import numpy as np
from basictdf import Tdf
from basictdf.tdfData3D import MarkerTrack

def add_marker_to_tdf(tdf_path, marker_name, coords):

    list_files = [tdf_path]

    for uri_file in list_files:
        with Tdf(uri_file) as f:
            walking_data3d = f.data3D
        if walking_data3d is None:return

        #print("\n",uri_file,"\n3D Antes")
        #print(walking_data3d)

        coords= np.array(coords)      
        
        trackData = np.ndarray((0, 3))
        time = []
        for x in walking_data3d["r asis"].data: # me copio de este marcador
            time = coords.tolist()
            trackData = np.vstack((trackData, time))
            time = []

        t2 = MarkerTrack(marker_name, trackData)
        walking_data3d.add_track(t2) 

        #print("\nNew\n")
        #print(walking_data3d[marker_name].data)

        with Tdf(uri_file).allow_write() as tdf:
            tdf.data3D = walking_data3d
        
            #print("\n3D Después")
            #print(tdf.data3D)



