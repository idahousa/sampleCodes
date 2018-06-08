import tensorflow as tf
import os


"""Reference link: https://www.tensorflow.org/api_docs/python/tf/gfile"""
"""Define some functions which uses the tf.gfile module"""
"""Using MkDir() to create a single directory. 
If, the dir_name contains some nested directory, we must use MakeDirs() instead"""
"""Use the tf.gfile.Exist(directory_name) to check whether the directory is existed or not!"""

def make_single_directory(directory_name):
    if not tf.gfile.Exists(directory_name):
        tf.gfile.MkDir(dirname=directory_name)
        return True
    else:
        return False

def make_nested_hierarchical_directory(directory_name):
    if not tf.gfile.Exists(directory_name):
        tf.gfile.MakeDirs(dirname=directory_name)
        return True
    else:
        return False

def rename_a_directory(old_name, new_name):
    if tf.gfile.Exists(old_name):
        if tf.gfile.Exists(new_name):
            return False
        else:
            tf.gfile.Rename(oldname=old_name, newname=new_name)
            return True
    else:
        return False

def delete_a_file(file_path):
    if tf.gfile.Exists(file_path):
        tf.gfile.Remove(file_path)
        return True
    else:
        return False

def copy_data_from_one_place_to_another(old_file_path,new_file_path):
    if not tf.gfile.IsDirectory(old_file_path):
        if not tf.gfile.Exists(new_file_path):
            tf.gfile.Copy(oldpath=old_file_path,newpath=new_file_path)
            return True
        else:
            return False
    else:
        return False

def get_list_of_files_in_directory(path_to_directory):
    if tf.gfile.Exists(path_to_directory):
        return tf.gfile.ListDirectory(path_to_directory)
    else:
        return []

"""Example code"""
if __name__ == '__main__':
    """1. Get the current directory to the working directory"""
    current_dir = os.getcwd()
    """Make a directory"""
    single_dir = current_dir + '/single_dir'
    nested_dir = current_dir + '/nested_directory/single_dir'
    print(make_single_directory(single_dir))
    print(make_nested_hierarchical_directory(nested_dir))
    """Rename a directory"""
    new_single_name = current_dir + '/single_directory'
    new_nested_name = current_dir + '/nested_directory/single_directory'
    print(rename_a_directory(single_dir,new_single_name))
    print(rename_a_directory(nested_dir,new_nested_name))
    """Get list of files in a directory"""
    file_list = get_list_of_files_in_directory(new_single_name)
    for i in file_list:
        print(i)
    """Copy and Delete a file"""
    delete_file_path = new_single_name + '/data.txt'
    reserved_file_path = new_single_name + '/new_data.txt'
    print(delete_file_path)
    print(reserved_file_path)
    print(copy_data_from_one_place_to_another(delete_file_path,reserved_file_path))
    """Delete a file"""
    print(delete_a_file(delete_file_path))
