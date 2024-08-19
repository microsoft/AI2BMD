import math
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

from AIMD import arguments, envflags
from utils.system import get_physical_core_count
from utils.utils import record_time
from Calculators.device_strategy import DeviceStrategy


def run_command(command: str, cwd_path: str) -> None:
    r"""
    Create a child process and run the command in the cwd_path.
    It is more safe than os.system.
    """

    if envflags.DEBUG_RC:
        print("run_command: ", command)

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd_path,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errorcode = proc.wait()
    if errorcode:
        path = cwd_path
        msg = (
            'Failed with command "{}" failed in '
            ""
            "{} with error code {}"
            "stdout: {}"
            "stderr: {}".format(command, path, errorcode, proc.stdout.read().decode(), proc.stderr.read().decode())
        )
        raise ValueError(msg)
    elif envflags.DEBUG_RC:
        print('-------------- stdout -----------------')
        print(proc.stdout.read().decode())
        print('-------------- stderr -----------------')
        print(proc.stderr.read().decode())


def run_command_mamba(command: str, cwd_path: str, mamba_env: str) -> None:
    command_with_env = f'bash -c "source  /etc/profile.d/source_conda.sh &&  mamba activate {mamba_env} && {command}"'
    run_command(command_with_env, cwd_path)


class Preprocess(object):
    r"""
    Preprocess the protein. Run the @method run_preprocess
    to start the preprocessing.

    Parameters:
    -----------
        prot_path: str
            The .pdb file path of the original protein.
            Both .pdb and removed .pdb version are supported.
            (/path/to/protein.pdb or /path/to/protein)
            Recommend the abs path.
        command_save_path: str
            The path to save the preprocess command results.
    """


    @staticmethod
    def get_seq_dict_path():
        return f"{arguments.get().log_dir}/seq_dict.pkl"


    def __init__(self, prot_path: str, utils_dir: str, command_save_path: str, solvent_method: str, log_dir: str, temp_k: float) -> None:
        self.prot_path = prot_path
        self.utils_dir = utils_dir
        self.command_save_path = command_save_path
        self.solvent_method = solvent_method
        if self.prot_path.endswith(".pdb"):
            self.prot_path = self.prot_path[:-4]  # remove the .pdb
        self.log_dir = log_dir
        self.temp_k = temp_k
        self.devices = DeviceStrategy.get_preprocess_device()

    def count_residues(self,top_file)-> int:
        with open(top_file, 'r') as file:
            lines = file.readlines()
        # Find the start and end of the RESIDUE_LABEL section
        for line in lines:
            if line.startswith('%FLAG RESIDUE_LABEL'):
                start = lines.index(line) + 2
                # print(line)
            if line.startswith('%FLAG RESIDUE_POINTER'):
                end = lines.index(line) - 1
                # print(line)
        # Count the residues
        num_residue = 0
        residue_lines = lines[start:end]
        for line in residue_lines:
            for word in line.split():
                if word.lower() not in {'na+','cl-','wat'}:
                    num_residue += 1
        self.num_residue = num_residue
        return num_residue

    def run_leap_mm(self) -> Tuple[str, str]:
        """
        Input: protein pdb
        Output: {prot_name}1.top {prot_name}1.inpcrd
        By-products:
            - t1.in t2.in 
            - (AMOEBA) {prot_name}_amobea.pdb {prot_name}.key
            - min1.in mm_a.in
        """
        with open("t1.in", "w") as fleap:
            fleap.write(
                "{}\n{}\n{}\n{}\n{}\n{}\n".format(
                    "source leaprc.protein.ff19SB",
                    "source leaprc.water.tip3p",
                    "mol = loadpdb {}.pdb".format(self.prot_path),
                    "solvatebox mol TIP3PBOX 10",
                    "saveamberparm mol {0}1.top {0}1.inpcrd".format(self.prot_path),
                    "quit",
                )
            )
        run_command_mamba("tleap -f t1.in", self.command_save_path, "ambertools")
        text = open(f"{self.prot_path}1.top").read()
        water_num = text.count("WAT")
        pos_num = (
            text.count(" ARG ")
            + text.count(" LYS ")
            + text.count(" HIS ")
            + text.count(" HID ")
            + text.count(" HIP ")
            + text.count(" HIE ")
        )
        neg_num = text.count(" ASP ") + text.count(" GLU ")
        ion_num = round(water_num * 0.002772)
        run_command("rm -rf leap.log", self.command_save_path)

        with open("t2.in", "w") as fleap:
            if neg_num >= pos_num:
                fleap.write(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                        "source leaprc.protein.ff19SB",
                        "loadamberparams frcmod.ionsjc_tip3p",
                        "source leaprc.water.tip3p",
                        "mol = loadpdb {}.pdb".format(self.prot_path),
                        "solvatebox mol TIP3PBOX 10",
                        f"addIons mol Na+ {ion_num+neg_num-pos_num}",
                        "addIons mol Cl- 0",
                        "saveamberparm mol {0}.top {0}.inpcrd".format(
                            self.prot_path
                        ),
                        "quit",
                    )
                )
            else:
                fleap.write(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                        "source leaprc.protein.ff19SB",
                        "source leaprc.water.tip3p",
                        "loadamberparams frcmod.ionsjc_tip3p",
                        "mol = loadpdb {}.pdb".format(self.prot_path),
                        "solvatebox mol TIP3PBOX 10",
                        "addIons mol Cl- 0",
                        f"addIons mol Na+ {ion_num}",
                        "addIons mol Cl- 0",
                        "saveamberparm mol {0}.top {0}.inpcrd".format(
                            self.prot_path
                        ),
                        "quit",
                    )
                )
        run_command_mamba("tleap -f t2.in", self.command_save_path, "ambertools")
        maxcyc = arguments.get().max_cyc
        ncyc = maxcyc // 2
        num_residue = self.count_residues(f"{self.prot_path}.top")
        
        if self.solvent_method == "AMOEBA":
            run_command_mamba(
                f"ambpdb -p {self.prot_path}.top -c {self.prot_path}.inpcrd > {self.prot_path}_amobea.pdb",
                self.command_save_path,
                "ambertools"
            )
            with open("leap.log") as f:
                text = f.read()
            pme = None
            for line in text.split("\n"):
                if "Total vdw box size" in line:
                    pme = round(
                        1.5
                        * max(
                            float(line.split()[4]),
                            float(line.split()[5]),
                            float(line.split()[6]),
                        )
                    )
                    break
            if pme is None:
                print("!!! pme is not extracted from log file")
                sys.exit(-1)
            for i in range(99999):
                if (
                    (pme + i) % 2 == 0
                    and (pme + i) % 3 == 0
                    and (pme + i) % 4 == 0
                    and (pme + i) % 5 == 0
                ):
                    pme = pme + i
                    break
            with open(f"{self.prot_path}.key", "w") as f:
                f.write(
                    f"""#
#  Keyfile for MD Benchmark on AMOEBA DHFR/Water Box
#

parameters            {self.utils_dir}/amoebapro13
#verbose
randomseed            123456789
neighbor-list

#
#  Define the Periodic Box and Cutoffs
#

a-axis                {pme}
b-axis                {pme}
c-axis                {pme}
vdw-cutoff            12.0

#
#  Set Parameters for Ewald Summation
#

ewald
ewald-cutoff          7.0
pme-grid              {pme} {pme} {pme}
fft-package           fftw
polar-eps             0.01

"""
                )
            with open("min1.in", "w") as fmin:
                fmin.write(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                        "Energy minimization",
                        "&cntrl",
                        " imin=1, !minimize the initial struc",
                        f" maxcyc={maxcyc}, !maximum number of cycles for minimization",
                        f" ncyc={ncyc}, !switch from steepest descent to conjugate",
                        " ntpr=1, ntwr=1,",
                        " ntb=1, !  Constant volume",
                        " ntp=0, !  No pressure scaling",
                        " ntf=1, ! complete force evaluation",
                        " ntc=1, ! No SHAKE",
                        " jfastw=4,  iamoeba=1,",
                        " /",
                        " ",
                        " &ewald",
                        f" nfft1={pme},nfft2={pme},nfft3={pme},",
                        "  skinnb=2.,nbtell=0,order=5,ew_coeff=0.5446",
                        " /",
                        " &amoeba",
                        "   dipole_scf_tol = 0.001,dipole_scf_iter_max=100,",
                        "   sor_coefficient=0.61,ee_damped_cut=4.5,ee_dsum_cut=6.7,",
                        " /",
                    )
                )

            with open("mm_a.in", "w") as f:
                f.write(
                    f"""zero step md to get energy and force
&cntrl
imin=0, nstlim=0,  ntx=1,
cut=10, ntb=1,  dt=0.001,
ntpr=1,ntwe=1,ntwx=1,ntwf=1,
jfastw=4,  iamoeba=1,
/
&ewald
nfft1={pme},nfft2={pme},nfft3={pme},
skinnb=2.,nbtell=0,order=5,ew_coeff=0.5446,
/
&amoeba
dipole_scf_tol = 0.001,dipole_scf_iter_max=100,
sor_coefficient=0.61,ee_damped_cut=4.5,ee_dsum_cut=6.7,
/

            """
                )
            run_command(f"cp {self.prot_path}_amobea.pdb test.pdb ", self.command_save_path)
            run_command(f"cp {self.prot_path}.key test.key", self.command_save_path)
            run_command(f"cp {self.prot_path}.top test.prmtop ", self.command_save_path)

            run_command(
                """awk '{ if($0 ~ /Na\+/) { gsub(/Na\+/, " NA", $0); gsub(/Na\+/, " NA", $0); gsub(/ATOM  /, "HETATM", $0); } print $0; }' test.pdb > test1.pdb""",
                self.command_save_path,
            )
            run_command(
                """awk '{ if($0 ~ /Cl\-/) { gsub(/Cl\-/, " CL", $0); gsub(/Cl\-/, " Cl", $0); gsub(/ATOM  /, "HETATM", $0); } print $0; }' test1.pdb > test2.pdb""",
                self.command_save_path,
            )
            run_command("mv test2.pdb test.pdb", self.command_save_path)
            run_command(
                f"""pdbxyz test << _EOF
{os.getcwd()}/test.key
_EOF""",
                self.command_save_path,
            )

            run_command(
                "analyze test PC > test.analout", self.command_save_path
            )
            run_command(
                "sed -i 's/Atom Type Definition Parameters/Atom Definition Parameters/g' test.analout",
                self.command_save_path,
            )
            run_command(
                "rm -rf test_tinker2amber.top test_tinker2amber.crd ",
                self.command_save_path,
            )
            run_command(
                """tinker_to_amber  -key test.key << _EOF
test.analout
test.xyz
test.pdb
test
test_tinker2amber.top
test_tinker2amber.crd
_EOF""",
                self.command_save_path,
            )
            run_command(
                f"cp test_tinker2amber.top {self.prot_path}.top", self.command_save_path
            )
            run_command(
                f"cp test_tinker2amber.crd {self.prot_path}.inpcrd", self.command_save_path
            )
            run_command_mamba(
                "ambpdb -p test_tinker2amber.top -c test_tinker2amber.crd > test_after.pdb",
                self.command_save_path,
                "ambertools"
            )
            return f"{self.prot_path}.top",f"{self.prot_path}.inpcrd"
        else:
            raise ValueError(f"Unrecognized solvent method: {self.solvent_method}")

    def run_emin_mm(self) -> str:
        """Run minimization with sander (AMBER), use this when there's no pre-equilibration with sander
        Input:  min1.in
        Output: preeq.rst
        By-products: min.out
        """
        run_command_mamba(
            "sander -O -i min1.in -p {0}.top -c {0}.inpcrd -o min.out "
            "-r preeq.rst -ref {0}.inpcrd".format(self.prot_path),
            self.command_save_path,
            "ambertools"
        )
        return f"{self.prot_path}.inpcrd"

    def generate_pdb_after_min(self) -> str:
        """Use cpptraj to convert sander output to pdb.
        Input: {prot_name}.top preeq.rst
        Output: {prot_name}-preeq.pdb
        """
        preeq_pdb = f"{self.prot_path}-preeq.pdb"
        with open("gene_pdb_from_rst.in", "w") as frst:
            frst.write(
                "{}\n{}\n{}\n".format(
                    "parm {}.top".format(self.prot_path),
                    "trajin preeq.rst",
                    "trajout {}".format(preeq_pdb),
                )
            )

        run_command_mamba("cpptraj -i gene_pdb_from_rst.in", self.command_save_path, "ambertools")

        return preeq_pdb

    def minimize1(self) -> None:
        """Sander minimization step 1.
        Input: min1.in {prot_name}.top {prot_name}.inpcrd
        Output: min1.rst
        By-products: min1.out
        """
        run_command_mamba(
            "sander -O -i min1.in -p {0}.top -c {0}.inpcrd -o min1.out -inf min1.info "
            "-r min1.rst -x min1.mdcrd -ref {0}.inpcrd".format(self.prot_path),
            self.command_save_path,
            "ambertools"
        )
        
    def minimize2(self) -> None:
        """Sander minimization step 2.
        Input: min1.rst {prot_name}.top {prot_name}.inpcrd
        Output: min2.rst
        By-products: min2.in min2.out
        """
        maxcyc = arguments.get().max_cyc
        ncyc = maxcyc // 2
        data = f"""Min2
                &cntrl
                imin=1,
                maxcyc={maxcyc},
                ncyc={ncyc},
                iwrap=1,
                ntb= 1,
                cut=10
                /
                """
        with open("min2.in", "w") as f:
            f.write(data)
        run_command_mamba(
            "sander -O -i min2.in -p {0}.top -c min1.rst -o min2.out -inf min2.info "
            "-r min2.rst -x min2.mdcrd -ref min1.rst".format(self.prot_path),
            self.command_save_path,
            "ambertools"
        )



    def generate_xyz_from_min(self) -> str:
        preeq_min_xyz = f"{self.prot_path}-preeq.xyz"
        with open("xyz.in", "w") as fxyz:
            fxyz.write(
                "{}\n{}\n{}\n{}".format(
                    "parm {}.top".format(self.prot_path),
                    "trajin preeq.rst",
                    "trajout {} ftype xyz prec 8".format(preeq_min_xyz),
                    "go",
                )
            )

        run_command_mamba("cpptraj -i xyz.in", self.command_save_path, "ambertools")
        return preeq_min_xyz


    def strip_wat(self, preeq_pdb) -> Tuple[str, str]:
        """Obtain 'nowat' pdb and topology files
        Input: {prot_name}-preeq.pdb
        Output: {prot_name}-preeq-nowat.pdb {prot_name}-preeq-nowat.top
        """

        preeq_nowat_top = f"{self.prot_path}-preeq-nowat.top"
        preeq_nowat_pdb = f"{self.prot_path}-preeq-nowat.pdb"

        with open("strip_wat_pdb.in", "w") as fwt:
            fwt.write(
                "{}\n{}\n{}\n{}\n".format(
                    "parm {}.top".format(self.prot_path),
                    "trajin {}".format(preeq_pdb),
                    "strip :WAT,Cl-,Na+,CL,NA",
                    "trajout {}".format(preeq_nowat_pdb),
                )
            )

        run_command_mamba("cpptraj -i strip_wat_pdb.in", self.command_save_path, "ambertools")
        if self.solvent_method == "AMOEBA":
            run_command(f"sed -i '/^TER/d' {preeq_nowat_pdb} ", self.command_save_path)
            run_command(f"cp {preeq_nowat_pdb} test1.pdb", self.command_save_path)
            run_command("cp test.key test1.key", self.command_save_path)
            run_command(
                f"""pdbxyz test1 << _EOF
{os.getcwd()}/test1.key
_EOF""",
                self.command_save_path,
            )
            run_command(
                "analyze test1 PC > test1.analout", self.command_save_path
            )
            run_command(
                "sed -i 's/Atom Type Definition Parameters/Atom Definition Parameters/g' test1.analout",
                self.command_save_path,
            )
            run_command(
                "rm -rf test1_tinker2amber.top test1_tinker2amber.crd ",
                self.command_save_path,
            )
            run_command(
                """tinker_to_amber  -key test1.key << _EOF
test1.analout
test1.xyz
test1.pdb
test1
test1_tinker2amber.top
test1_tinker2amber.crd
_EOF""",
                self.command_save_path,
            )
            run_command(
                f"cp test1_tinker2amber.top {preeq_nowat_top}", self.command_save_path
            )
        else:
            raise ValueError(f"Unrecognized solvent method: {self.solvent_method}")

        return preeq_nowat_top, preeq_nowat_pdb

    def organize_files(self, file_list: List[str], solvent_method: str) -> List[str]:
        r"""
        Move the generated files to a unified folder.
        """
        mm_path = os.path.join(
            os.path.dirname(self.prot_path),
            f"{os.path.basename(self.prot_path)}_mm",
        )
        os.makedirs(mm_path, exist_ok=True)
        moved_file_list = []

        for file in file_list:
            shutil.copy(file, mm_path)
            moved_file_list.append(
                os.path.join(mm_path, os.path.basename(file))
            )

        return moved_file_list

    def check_exist(self, solvent_method: str):
        mm_path = os.path.join(
            os.path.dirname(self.prot_path),
            f"{os.path.basename(self.prot_path)}_mm",
        )
            
        if os.path.exists(mm_path) and os.listdir(mm_path):
            preeq_top = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}.top"
            )
            preeq_inpcrd = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}.inpcrd"
            )
            preeq_pdb = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}-preeq.pdb"
            )
            preeq_nowat_top = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}-preeq-nowat.top"
            )
            preeq_nowat_pdb = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}-preeq-nowat.pdb"
            )
            mm_a = os.path.join(mm_path, "mm_a.in")
            if {
                preeq_top,
                preeq_inpcrd,
                preeq_pdb,
                preeq_nowat_top,
                preeq_nowat_pdb,
                mm_a,
            } != set([os.path.join(mm_path, p) for p in os.listdir(mm_path)]):
                exist = set(
                    [
                        os.path.join(os.path.dirname(mm_path), p)
                        for p in os.listdir(mm_path)
                    ]
                )
                print(f"existing files: {exist}")
                expect = {
                    preeq_top,
                    preeq_inpcrd,
                    preeq_pdb,
                    preeq_nowat_top,
                    preeq_nowat_pdb,
                }
                print(f"expected files: {expect}")
                print(
                    "Some files missing, delete this folder and "
                    "rerun the preprocessing step..."
                )
                shutil.rmtree(mm_path)
                return False
            return (
                preeq_top,
                preeq_inpcrd,
                preeq_pdb,
                preeq_nowat_top,
                preeq_nowat_pdb,
                mm_a,
            )
        else:
            return False

    @record_time
    def run_preprocess(self) -> List[str]:
        os.makedirs(self.command_save_path, exist_ok=True)
        os.system(f"cp {self.utils_dir}/seq_dict_{self.solvent_method}.pkl {Preprocess.get_seq_dict_path()}")

        os.chdir(self.command_save_path)
        out = self.check_exist(self.solvent_method)
        if out:
            print("Preprocessing step already done, skip...")
            return list(out)
        # generate the topology file
        preeq_top, preeq_inpcrd = self.run_leap_mm()
        # run minimize & preeq 
        if self.solvent_method == 'AMOEBA':
            preeq_inpcrd = self.run_emin_mm()
        else:
            print('solvent method not supported')
            sys.exit(-1)
        preeq_pdb = self.generate_pdb_after_min()
        preeq_nowat_top, preeq_nowat_pdb = self.strip_wat(preeq_pdb)
        mm_in = self.command_save_path + "/" + "mm_a.in"
        # collect the generated files
        return self.organize_files(
            [
                preeq_top,
                preeq_inpcrd,
                preeq_pdb,
                preeq_nowat_top,
                preeq_nowat_pdb,
                mm_in,
            ],self.solvent_method
        )
