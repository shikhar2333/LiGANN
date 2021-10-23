'''
creates types and gninatypes files of the protein for input to CNN via libmolgrid
first argument is path to protein pocket_pdb_file
second argument is path to barycenters list pocket_pdb_file
'''
import molgrid
import struct
import numpy as np
import os
import sys

def gninatype(dir):
    # creates gninatype pocket_pdb_file for model input
    # j = 0
    for protein_code in os.listdir(dir):
        # print(j,protein_code)
        pocket_pdb_file = protein_code+"/"+protein_code+"_pocket.pdb"
        f=open(dir+"/"+pocket_pdb_file.replace('.pdb','.types'),'w')
        f.write(dir+"/"+pocket_pdb_file)
        f.close()
        atom_map=molgrid.FileMappedGninaTyper('gninamap')
        dataloader=molgrid.ExampleProvider(atom_map,shuffle=False,default_batch_size=1)
        train_types=pocket_pdb_file.replace('.pdb','.types')
        dataloader.populate(dir+"/"+train_types)
        example=dataloader.next()
        coords=example.coord_sets[0].coords.tonumpy()
        types=example.coord_sets[0].type_index.tonumpy()
        types=np.int_(types) 
        fout=open(dir+"/"+pocket_pdb_file.replace('.pdb','.gninatypes'),'wb')
        for i in range(coords.shape[0]):
            print(i+1,coords[i][0],coords[i][1],coords[i][2],types[i])
            fout.write(struct.pack('fffi',coords[i][0],coords[i][1],coords[i][2],types[i]))
        fout.close()
        os.remove(dir+"/"+train_types)
    # return pocket_pdb_file.replace('.pdb','.gninatypes')

def create_types(pocket_pdb_file,protein):
    # create types pocket_pdb_file for model predictions
    fout=open(pocket_pdb_file.replace('.txt','.types'),'w')
    fin =open(pocket_pdb_file,'r')
    for line in fin:
        fout.write(' '.join(line.split()) + ' ' + protein +'\n')
    return pocket_pdb_file.replace('.txt','.types')


if __name__ == '__main__':
    protein=gninatype(sys.argv[1])
#    types=create_types(sys.argv[2],protein)
