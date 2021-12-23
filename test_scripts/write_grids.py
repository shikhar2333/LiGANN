import molgrid
import torch

def get_gmaker_eproviders():
    e = molgrid.ExampleProvider(molgrid.defaultGninaReceptorTyper,
            data_root="/scratch/shubham/v2020-other-PL",
            cache_structs=False, shuffle=True, stratify_receptor=True)
    fname = "/scratch/shubham/training_example_pocket.types"         
    e.populate(fname)
    m = molgrid.MolDataset(fname, data_root="/scratch/shubham/v2020-other-PL",
            typers=(molgrid.defaultGninaReceptorTyper,),
            make_vector_types=False)
    torch_loader = torch.utils.data.DataLoader(m, batch_size=8,
            collate_fn=molgrid.MolDataset.collateMolDataset)
    gmaker = molgrid.GridMaker()
    shape = gmaker.grid_dimensions(e.num_types())
    mgrid = molgrid.MGrid5f(8, *shape)
    for (i, batch) in enumerate(torch_loader):
        center, coords, types, radii = batch[1:5]
        gmaker.forward(center, coords, types, radii, mgrid.cpu())
        if i == 10:
            break

#    dims = gmaker.grid_dimensions(e.num_types())
    return m, gmaker

def write_grids(input_tensor, mol_batch):
    center = mol_batch.coord_sets[0].center()
#    print(center[0],center[1],center[2])
    for i in range(14):
        grid = molgrid.Grid3f(input_tensor[i].cpu())
        molgrid.write_dx("grid_"+str(i)+".dx", grid, center, 0.5, 1)


if __name__ == "__main__":
    e, gmaker = get_gmaker_eproviders()
#    print("4D tensor shape: ", dims)
#    input_tensor = torch.zeros(dims, dtype=torch.float32, device='cuda')
#    for _ in range(1):
#        mol_batch = e.next()
#        gmaker.forward(mol_batch, input_tensor, 0)
#        print("src file", mol_batch.coord_sets[0].src)
#        write_grids(input_tensor, mol_batch)
