import math
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

from AIMD import arguments, envflags
from Calculators.device_strategy import DeviceStrategy
from utils.utils import record_time, reorder_atoms


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
                if word.lower() not in {'na', 'na+', 'cl', 'cl-', 'wat'}:
                    num_residue += 1
        self.num_residue = num_residue
        return num_residue

    def run_leap_mm(self) -> Tuple[str, str]:
        """
        Input: protein pdb
        Output: (AMOEBA) {prot_name}-preeq.pdb {prot_name}-preeq-nowat.pdb
        By-products:
            - t1.in t2.in
            - (AMOEBA) convert.in min1.key, min2.key
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

        if self.solvent_method == "AMOEBA":
            if self.devices.startswith("cuda"):
                self.amoeba_command_dir = '/usr/local/gpu-m'
            else:
                self.amoeba_command_dir = '/usr/local/cpu-m'

            random_seed = 23

            # generate pdb from top and inpcrd
            with open("convert_pdb.in", "w") as fcpptraj:
                fcpptraj.write("{}\n{}\n{}\n".format(
                    "parm {}.top".format(self.prot_path),
                    "trajin {}.inpcrd".format(self.prot_path),
                    "trajout {}_tleap.pdb".format(self.prot_path)
                ))
            run_command_mamba("cpptraj -i convert_pdb.in", self.command_save_path, "ambertools")

            # get pbc and solute atom number
            with open(f"{self.prot_path}_tleap.pdb") as f:
                text = f.readlines()
                head = text[0].split()
                pbc_x = math.ceil(float(head[1]))
                pbc_y = math.ceil(float(head[2]))
                pbc_z = math.ceil(float(head[3]))
                solute_atom_num = len([x for x in text if x.startswith("ATOM") and x[17:20].strip().lower() not in ["na", "na+", "cl", "cl-", "wat"]])

            # convert pdb to tinker amoeba xyz
            run_command(f"{self.amoeba_command_dir}/pdbxyz8 {self.prot_path}_tleap.pdb {self.utils_dir}/amoebabio18.prm", self.command_save_path)

            # write key file for minimization 1
            with open("min1.key", "w") as f:
                f.write("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                    f"parameters {self.utils_dir}/amoebabio18.prm",
                    f"randomseed {random_seed}",
                    f"a-axis {pbc_x}",
                    f"b-axis {pbc_y}",
                    f"c-axis {pbc_z}",
                    "cutoff 12",
                    "vdw-cutoff 12",
                    f"restrain-position -1 {solute_atom_num} 1000.0 0.0",
                    "ewald",
                    "ewald-cutoff 7.0",
                    "fft-package FFTW",
                    "polarization mutual",
                    "polar-eps 0.01",
                    "minimize",
                    f"maxiter {maxcyc}"
                ))

            run_command(f"{self.amoeba_command_dir}/minimize9 {self.prot_path}_tleap.xyz 0.1 -k min1.key", self.command_save_path)

            # write key file for minimization 2
            with open("min2.key", "w") as f:
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

            run_command(f"{self.amoeba_command_dir}/minimize9 {self.prot_path}_tleap.xyz_2 0.1 -k min2.key", self.command_save_path)

            # convert xyz to pdb
            run_command(f"{self.amoeba_command_dir}/xyzpdb8 {self.prot_path}_tleap.xyz_3 {self.utils_dir}/amoebabio18.prm", self.command_save_path)
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
        else:
            raise ValueError(f"Unrecognized solvent method: {self.solvent_method}")


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
            preeq_pdb = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}-preeq.pdb"
            )
            preeq_nowat_pdb = os.path.join(
                mm_path, f"{os.path.basename(self.prot_path)}-preeq-nowat.pdb"
            )

            if solvent_method == 'AMOEBA':
                exist = {
                    os.path.join(mm_path, p)
                    for p in os.listdir(mm_path)
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
                    shutil.rmtree(mm_path)
                    return False
                return (preeq_pdb,
                        preeq_nowat_pdb,
                )
            else:
                raise ValueError(f"Unrecognized solvent method: {solvent_method}")
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

        if self.solvent_method == 'AMOEBA':
            preeq_pdb, preeq_nowat_pdb = self.run_leap_mm()

            reorder_atoms(preeq_pdb)
            reorder_atoms(preeq_nowat_pdb)

            return self.organize_files(
                [
                    preeq_pdb,
                    preeq_nowat_pdb,
                ],
                self.solvent_method,
            )
        else:
            print('solvent method not supported')
            sys.exit(-1)
