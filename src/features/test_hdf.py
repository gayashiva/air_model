"""
Created on Tue Dec 10 21:26:54 2019

@author: Christopher J. Burke
Give a worked example of saving a list of class objects with mixed
storage types to a HDF5 file and reading in file back to a list of class
objects.  The solution is inspired by this bug report
https://github.com/h5py/h5py/issues/735
and the numpy and hdf5 documentation
"""

import numpy as np
import h5py

class test_object:
    """ Define a storage class that keeps info that we want to record
      for every object
    """
    # explictly state the name, datatype and shape for every
    #  class variable
    #  The names MUST exactly match the class variable names in the __init__
    store_names = ['a', 'b', 'c', 'd', 'e']
    store_types = ['i8', 'i4', 'f8', 'S80', 'f8']
    store_shapes = [None, None, None, None, [4]]
    # Make the tuples that will define the numpy structured array
    # https://docs.scipy.org/doc/numpy/user/basics.rec.html
    sz = len(store_names)
    store_def_tuples = []
    for i in range(sz):
        if store_shapes[i] is not None:
            store_def_tuples.append((store_names[i], store_types[i], store_shapes[i]))
        else:
            store_def_tuples.append((store_names[i], store_types[i]))
    # Actually define the numpy structured/compound data type
    store_struct_numpy_dtype = np.dtype(store_def_tuples)

    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0.0
        self.d = '0'
        self.e = [0.0, 0.0, 0.0, 0.0]

    def store_objlist_as_hd5f(self, objlist, fileName):
        """Function to save the class structure into hdf5
        objlist -  is a list of the test_objects
        fileName - is the h5 filename for output
        """        
        # First create the array of numpy structered arrays
        np_dset = np.ndarray(len(objlist), dtype=self.store_struct_numpy_dtype)
        # Convert the class variables into the numpy structured dtype
        for i, curobj in enumerate(objlist):
            for j in range(len(self.store_names)):
                np_dset[i][self.store_names[j]] = getattr(curobj, self.store_names[j])
        # Data set should be all loaded ready to write out
        fp = h5py.File(fileName, 'w')
        hf_dset = fp.create_dataset('dset', shape=(len(objlist),), dtype=self.store_struct_numpy_dtype)
        hf_dset[:] = np_dset
        fp.close()

    def fill_objlist_from_hd5f(self, fileName):
        """ Function to read in the hdf5 file created by store_objlist_as_hdf5
          and store the contents into a list of test_objects
          fileName - si the h5 filename for input
         """
        fp = h5py.File(fileName, 'r')
        np_dset = np.array(fp['dset'])
        # Start with empty list
        all_objs = []
        # iterate through the numpy structured array and save to objects
        for i in range(len(np_dset)):
            tmp = test_object()
            for j in range(len(self.store_names)):
                setattr(tmp, self.store_names[j], np_dset[i][self.store_names[j]])
            # Append object to list
            all_objs.append(tmp)
        return all_objs

if __name__ == '__main__':

    all_objs = []    
    for i in range(3):
        # instantiate tce_seed object
        tmp = test_object()
        # Put in some dummy data into object
        tmp.a = int(i)
        tmp.b = int(i)
        tmp.c = float(i)
        tmp.d = '{0} {0} {0} {0}'.format(i)
        tmp.e = np.full([4], i, dtype=np.float)
        all_objs.append(tmp)

    # Write out hd5 file
    tmp.store_objlist_as_hd5f(all_objs, 'test_write.h5')

    # Read in hd5 file
    all_objs = []
    all_objs = tmp.fill_objlist_from_hd5f('test_write.h5')

    # verify the output is as expected
    for i, curobj in enumerate(all_objs):
        print('Object {0:d}'.format(i))
        print('{0:d} {1:d} {2:f}'.format(curobj.a, curobj.b, curobj.c))
        print('{0} {1}'.format(curobj.d.decode('ASCII'), curobj.e))
