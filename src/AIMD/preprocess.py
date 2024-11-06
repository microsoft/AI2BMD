import math
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

from AIMD import arguments, envflags
from Calculators.device_strategy import DeviceStrategy
from utils.pdb import reorder_atoms, standardise_pdb, translate_coord_pdb, reorder_coord_amber2tinker
from utils.system import get_physical_core_count
from utils.utils import record_time


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
        text=True,
    )
    out, err = proc.communicate()
    if proc.returncode:
        path = cwd_path
        msg = (
            'Failed with command "{}" failed in '
            ""
            "{} with error code {}"
            "stdout: {}"
            "stderr: {}".format(command, path, proc.returncode, out, err)
        )
        raise ValueError(msg)
    elif envflags.DEBUG_RC:
        print('-------------- stdout -----------------')
        print(out)
        print('-------------- stderr -----------------')
        print(err)


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

    def __init__(self, prot_path: str, utils_dir: str, command_save_path: str, preprocess_method: str, log_dir: str, temp_k: float) -> None:
        self.prot_path = prot_path
        self.utils_dir = utils_dir
        self.command_save_path = command_save_path
        self.preprocess_method = preprocess_method
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
                if word.lower() not in {'na', 'na+', 'cl', 'cl-', 'wat'}:
                    num_residue += 1
        self.num_residue = num_residue
        return num_residue

    def run_leap_mm(self) -> Tuple[str, str]:
        """
        Input: protein pdb
        Output: (FF19SB) {prot_name}1.top {prot_name}1.inpcrd
                (AMOEBA) {prot_name}-preeq.pdb {prot_name}-preeq-nowat.pdb
        """
        with open("t1.in", "w") as fleap:
            fleap.write(
                "{}\n{}\n{}\n{}\n{}\n{}\n".format(
                    "source leaprc.protein.ff19SB",
                    "source leaprc.water.tip3p",
                    "mol = loadpdb {}.pdb".format(self.prot_path),
                    "solvatebox mol TIP3PBOX 20",
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
                        "solvatebox mol TIP3PBOX 20",
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
                        "solvatebox mol TIP3PBOX 20",
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

        if self.preprocess_method == "FF19SB":
            return f"{self.prot_path}.top",f"{self.prot_path}.inpcrd"

        if self.preprocess_method == "AMOEBA":
            if self.devices.startswith("cuda"):
                self.amoeba_command_dir = '/usr/local/gpu-m'
            else:
                self.amoeba_command_dir = '/usr/local/cpu-m'

            random_seed = arguments.get().seed

            # generate pdb from top and inpcrd
            with open("convert_pdb.in", "w") as fcpptraj:
                fcpptraj.write("{}\n{}\n{}\n".format(
                    "parm {}.top".format(self.prot_path),
                    "trajin {}.inpcrd".format(self.prot_path),
                    "trajout {}_tleap1.pdb".format(self.prot_path)
                ))
            run_command_mamba("cpptraj -i convert_pdb.in", self.command_save_path, "ambertools")

            # translate pdb to center
            pbc_x, pbc_y, pbc_z = translate_coord_pdb(f"{self.prot_path}_tleap1.pdb", f"{self.prot_path}_tleap.pdb")

            # convert pdb to tinker amoeba xyz
            run_command(f"{self.amoeba_command_dir}/pdbxyz8 {self.prot_path}_tleap.pdb {self.utils_dir}/amoebabio18.prm", self.command_save_path)

            # write key file for minimization
            with open("min.key", "w") as f:
                f.write("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                    f"parameters {self.utils_dir}/amoebabio18.prm",
                    f"randomseed {random_seed}",
                    f"a-axis {pbc_x}",
                    f"b-axis {pbc_y}",
                    f"c-axis {pbc_z}",
                    f"cutoff 12",
                    f"vdw-cutoff 12",
                    "ewald",
                    "ewald-cutoff 7.0",
                    "fft-package FFTW",
                    "polarization mutual",
                    "polar-eps 0.01",
                    "minimize",
                    f"maxiter {maxcyc}"
                ))

            run_command(f"{self.amoeba_command_dir}/minimize9 {self.prot_path}_tleap.xyz 0.1 -k min.key", self.command_save_path)

            # convert xyz to pdb
            run_command(f"{self.amoeba_command_dir}/xyzpdb8 {self.prot_path}_tleap.xyz_2 {self.utils_dir}/amoebabio18.prm", self.command_save_path)
            run_command(f"cp {self.prot_path}_tleap.pdb_2 {self.prot_path}-preeq.pdb", self.command_save_path)

            # get solute only pdb， remove water and ions
            with open(f"{self.prot_path}-preeq.pdb") as f:
                text = f.readlines()
                with open(f"{self.prot_path}-preeq-nowat.pdb", "w") as fsolute:
                    atom_begidx = 0
                    for line in text:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            break
                        else:
                            atom_begidx += 1
                    fsolute.write("".join(text[:atom_begidx]))
                    fsolute.write("".join([x for x in text if (x.startswith("ATOM") or x.startswith('HETATM')) and x[17:20].strip().lower() not in ["na", "na+", "cl", "cl-", "wat", "hoh"]]))
            return f"{self.prot_path}-preeq.pdb", f"{self.prot_path}-preeq-nowat.pdb"


    def preprocess_ff19sb(self, solv_top: str, solv_inpcrd: str) -> Tuple[str, str]:
        maxcyc = arguments.get().max_cyc
        ncyc = maxcyc // 2
        num_residue = self.count_residues(solv_top)

        num_process = get_physical_core_count()

        with open("min.in", "w") as fmin:
            fmin.write(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                    "Energy minimization",
                    "&cntrl",
                    " imin=1",
                    f" maxcyc={maxcyc}",
                    f" ncyc={ncyc}",
                    " iwrap = 1",
                    " cut=10.0",
                    " ntb=1",
                    " /"
                )
            )

        run_command_mamba(
            f"sander -O -i min.in -p {solv_top} -c {solv_inpcrd} -o min.out -inf min.info "
            f"-r min.rst -x min.mdcrd -ref {solv_inpcrd}", self.command_save_path, "ambertools"
        )

        """Sander heat steps for sander-based pre-equilibration.
        Input: min.rst {prot_name}.top {prot_name}.inpcrd
        Output: heat.rst
        By-products: heat.in heat.out
        """
        nstlim = 20000
        data = f"""heating from 0K too {self.temp_k}K
&cntrl
 imin = 0,
 irest = 0,
 ntx = 1,
 ntb = 1,
 iwrap = 1,
 cut = 10,
 ntr = 1,
 ntc = 2,
 ntf = 2,
 tempi = 0.0,
 temp0 = {self.temp_k},
 ntt = 3,vlimit = 10,
 gamma_ln = 1.0,
 nstlim = {nstlim}, dt = 0.002,
 ntpr = 1000, ntwx = 1000, ntwr = 1000
 /
Hold the protein fixed
10.0
RES 1 {num_residue}
END
END
"""

        with open("heat.in", "w") as f:
            f.write(data)

        run_command_mamba(
            f"sander -O -i heat.in -p {solv_top} -c min.rst -o heat.out -inf heat.info "
            "-r heat.rst -x heat.mdcrd -ref min.rst", self.command_save_path, "ambertools"
        )

        """Sander-based pre-equilibration stage 1.
        Input: heat.rst {prot_name}.top {prot_name}.inpcrd
        Output: preeq1.rst
        By-products: preeq1.in preeq1.out
        """
        nstlim = 20000
        data = f"""pre-eq1, NVT
&cntrl
 imin=0,
 ntb=1,
 ntp=0,
 iwrap=1,
 ntx=5,
 irest=1,
 ntr=1,
 ig=-1,
 ntc=1,
 ntf=1,
 ntpr=1000,
 ntwx=1000,
 ntwr=1000,
 ntt=3,
 gamma_ln=1.0,
 nstlim={nstlim},
 dt=0.001,
 cut=10.0
 tempi={self.temp_k},
 temp0={self.temp_k},
 /
Hold protein fixed
10.0
RES 1 {num_residue}
END
END
"""

        with open("preeq1.in", "w") as f:
            f.write(data)

        run_command_mamba(
            f"sander -O -i preeq1.in -c heat.rst -o preeq1.out -inf preeq1.info "
            f"-r preeq1.rst -x preeq1.mdcrd  -ref heat.rst -p {solv_top}", self.command_save_path, "ambertools"
        )

        """Sander-based pre-equilibration stage 2.
        Input: preeq1.rst {prot_name}.top {prot_name}.inpcrd
        Output: preeq2.rst
        By-products: preeq2.in preeq2.out
        """

        nstlim = 20000
        data = f"""pre-eq2, NVT
&cntrl
 imin=0,
 ntb=1,
 ntp=0,
 iwrap=1,
 ntx=5,
 irest=1,
 ntr=1,
 ig=-1,
 ntc=1,
 ntf=1,
 ntpr=1000,
 ntwx=1000,
 ntwr=1000,
 ntt=3,
 gamma_ln=1.0,
 nstlim={nstlim},
 dt=0.001,
 cut=10.0,
 tempi={self.temp_k},
 temp0={self.temp_k},
/
Hold protein fixed
10.0
RES :1-{num_residue}@CA
END
END
"""

        with open("preeq2.in", "w") as f:
            f.write(data)

        run_command_mamba(
            f"sander -O -i preeq2.in -c preeq1.rst -o preeq2.out -inf preeq2.info "
            f"-r preeq2.rst -x preeq2.mdcrd  -ref preeq1.rst -p {solv_top}", self.command_save_path, "ambertools"
        )

        """Sander-based pre-equilibration stage 3.
        Input: preeq2.rst {prot_name}.top {prot_name}.inpcrd
        Output: preeq3.rst
        By-products: preeq3.in preeq3.out
        """
        nstlim = 20000
        data = f"""pre-eq3, NVT
&cntrl
 imin=0,
 ntb=1,
 ntp=0,
 iwrap=1,
 ntx=5,
 irest=1,
 ig=-1,
 ntc=1,
 ntf=1,
 ntpr=1000,
 ntwx=1000,
 ntwr=1000,
 ntt=3,
 gamma_ln=1.0,
 nstlim={nstlim},
 dt=0.001,
 cut=10.0,
 tempi={self.temp_k},
 temp0={self.temp_k},
/
"""

        with open("preeq3.in", "w") as f:
            f.write(data)

        run_command_mamba(
            f"sander -O -i preeq3.in -c preeq2.rst -o preeq3.out -inf preeq3.info "
            f"-r preeq3.rst -x preeq3.mdcrd  -ref preeq2.rst -p {solv_top}", self.command_save_path, "ambertools"
        )

        """Sander-based pre-equilibration stage 4.
        Input: preeq3.rst {prot_name}.top {prot_name}.inpcrd
        Output: preeq.rst <- NOTE: different naming scheme than previous steps!
        By-products: preeq4.in preeq4.out
        """

        nstlim = 100000
        data = f"""pre-eq4, NPT
&cntrl
 imin=0,
 ntb=2,
 ntp=2,
 iwrap=1,
 ntx=5,
 irest=1,
 ig=-1,
 ntc=1,
 ntf=1,
 ntpr=1000,
 ntwx=1000,
 ntwr=1000,
 ntt=3,
 gamma_ln=1.0,
 nstlim={nstlim},
 dt=0.001,
 ioutfm=1,
 ntxo=2,
 cut=10,
 tempi={self.temp_k},
 temp0={self.temp_k},
/
"""

        with open("preeq4.in", "w") as f:
            f.write(data)

        run_command_mamba(
            f"sander -O -i preeq4.in -c preeq3.rst -o preeq4.out -inf preeq4.info "
            f"-r preeq.rst -x preeq4.mdcrd  -ref preeq3.rst -p {solv_top}", self.command_save_path, "ambertools"
        )

        # generate_pdb
        preeq_pdb = f"{self.prot_path}-preeq.pdb"
        with open("gene_pdb_from_rst.in", "w") as frst:
            frst.write(
                "{}\n{}\n{}\n".format(
                    f"parm {solv_top}",
                    f"trajin preeq.rst",
                    "trajout {}".format(preeq_pdb),
                )
            )

        run_command_mamba("cpptraj -i gene_pdb_from_rst.in", self.command_save_path, "ambertools")

        # get solute only pdb，remove water and ions
        with open(f"{self.prot_path}-preeq.pdb") as f:
            text = f.readlines()
            with open(f"{self.prot_path}-preeq-nowat.pdb", "w") as fsolute:
                atom_begidx = 0
                for line in text:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        break
                    else:
                        atom_begidx += 1
                fsolute.write("".join(text[:atom_begidx]))
                fsolute.write("".join([x for x in text if (x.startswith("ATOM") or x.startswith('HETATM')) and x[17:20].strip().lower() not in ["na", "na+", "cl", "cl-", "wat", "hoh"]]))

        return f"{self.prot_path}-preeq.pdb", f"{self.prot_path}-preeq-nowat.pdb"


    def organize_files(self, file_list: List[str]) -> List[str]:
        r"""
        Move the generated files to a unified folder.
        """
        preprocess_path = os.path.join(
            os.path.dirname(self.prot_path),
            f"{os.path.basename(self.prot_path)}_preprocessed",
        )
        os.makedirs(preprocess_path, exist_ok=True)
        moved_file_list = []

        for file in file_list:
            shutil.copy(file, preprocess_path)
            moved_file_list.append(
                os.path.join(preprocess_path, os.path.basename(file))
            )

        return moved_file_list

    def check_exist(self, preprocess_method: str):
        preprocess_path = os.path.join(
            os.path.dirname(self.prot_path),
            f"{os.path.basename(self.prot_path)}_preprocessed",
        )

        if not os.path.exists(preprocess_path) or not os.listdir(preprocess_path):
            return False

        preeq_pdb = os.path.join(
            preprocess_path, f"{os.path.basename(self.prot_path)}-preeq.pdb"
        )
        preeq_nowat_pdb = os.path.join(
            preprocess_path, f"{os.path.basename(self.prot_path)}-preeq-nowat.pdb"
        )

        exist = {
            os.path.join(preprocess_path, p)
            for p in os.listdir(preprocess_path)
        }
        expect = {
            preeq_pdb,
            preeq_nowat_pdb,
        }
        if exist != expect:
            print(f"existing files: {exist}")
            print(f"expected files: {expect}")
            print(
                "Some files missing, delete this folder and "
                "rerun the preprocessing step..."
            )
            shutil.rmtree(preprocess_path)
            return False

        return (preeq_pdb,
                preeq_nowat_pdb,
        )

    @record_time
    def run_preprocess(self) -> List[str]:
        os.makedirs(self.command_save_path, exist_ok=True)
        os.system(f"cp {self.utils_dir}/seq_dict.pkl {Preprocess.get_seq_dict_path()}")

        os.chdir(self.command_save_path)
        out = self.check_exist(self.preprocess_method)
        if out:
            print("Preprocessing step already done, skip...")
            return list(out)

        if self.preprocess_method == 'AMOEBA':
            preeq_pdb, preeq_nowat_pdb = self.run_leap_mm()

            reorder_atoms(preeq_pdb)
            reorder_atoms(preeq_nowat_pdb)

            standardise_pdb(preeq_pdb)
            standardise_pdb(preeq_nowat_pdb)

            return self.organize_files([preeq_pdb, preeq_nowat_pdb])

        if self.preprocess_method == 'FF19SB':
            solv_top, solv_inpcrd = self.run_leap_mm()
            preeq_pdb, preeq_nowat_pdb = self.preprocess_ff19sb(solv_top, solv_inpcrd)

            reorder_coord_amber2tinker(preeq_pdb)
            reorder_coord_amber2tinker(preeq_nowat_pdb)

            return self.organize_files([preeq_pdb, preeq_nowat_pdb])
