from utils.vaspUtils import VaspData
import numpy as np
if __name__ == '__main__':
    # vd = VaspData('file_lib/vasprun.xml')
    #
    # vd.run()
    # np.save("file_lib/incar_dict.npy",vd.incar_dict)
    # np.save("file_lib/atomtype_info_list.npy",vd.atomtype_info_list)
    # np.save("file_lib/position_list.npy",vd.position_list)
    # np.save("file_lib/track_poscar_info_list.npy",vd.track_poscar_info_list)
    # np.save("file_lib/track_energy_list.npy",vd.track_energy_list)
    pos = np.load("file_lib/position_list.npy")
    print(pos[0])
