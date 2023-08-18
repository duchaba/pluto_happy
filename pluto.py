# [BEGIN OF pluto_happy]
## required lib, required "pip install"
import torch
import cryptography
import cryptography.fernet
from flopth import flopth
import huggingface_hub
import huggingface_hub.hf_api
## standard libs, no need to install
import json
import requests
import time
import os
import random
import re
import sys
import psutil
import threading
import socket
import PIL
import pandas
import matplotlib
import numpy
import importlib.metadata
import types
import cpuinfo
import pynvml
import pathlib
import re
import subprocess
# define class Pluto_Happy
class Pluto_Happy(object):
  """
  The Pluto projects starts with fun AI hackings and become a part of my
  first book "Data Augmentation with Python" with Packt Publishing.

  In particular, Pluto_Happy is a clean and lite kernel of a simple class,
  and using @add_module decoractor to add in specific methods to be a new class,
  such as Pluto_HFace with a lot more function on HuggingFace, LLM and Transformers.

  Args:
      name (str): the display name, e.g. "Hanna the seeker"

  Returns:
      (object): the class instance.
  """

  # initialize the object
  def __init__(self, name="Pluto",*args, **kwargs):
    super(Pluto_Happy, self).__init__(*args, **kwargs)
    self.author = "Duc Haba"
    self.name = name
    self._ph()
    self._pp("Hello from class", str(self.__class__) + " Class: " + str(self.__class__.__name__))
    self._pp("Code name", self.name)
    self._pp("Author is", self.author)
    self._ph()
    #
    # define class var for stable division
    self._huggingface_crkey="gAAAAABkduT-XeiYtD41bzjLtwsLCe9y1FbHH6wZkOZwvLwCrgmOtNsFUPWVqMVG8MumazFhiUZy91mWEnLDLCFw3eKNWtOboIyON6yu4lctn6RCQ4Y9nJvx8wPyOnkzt7dm5OISgFcm"
    self._gpt_crkey="'gAAAAABkgiYGQY8ef5y192LpNgrAAZVCP3bo2za9iWSZzkyOJtc6wykLwGjFjxKFpsNryMgEhCATJSonslooNSBJFM3OcnVBz4jj_lyXPQABOCsOWqZm6W9nrZYTZkJ0uWAAGJV2B8uzQ13QZgI7VCZ12j8Q7WfrIg=='"
    self._fkey="your_key_goes_here"
    self._github_crkey="gAAAAABksjLYjRoFxZDDW5RgBN_uvm6pqDP128S2qOEfv9PgVL8fwdtXzWvCeMHwnGcibAky5cGs3XNxMH4VgbaPBA3I_CPRp3bRK3TMNU4HGRKxbdMnJ7U04IkVSdcMn8o86z3yhcSn"
    self._kaggle_crkey="gAAAAABksjOOU2a-BtZ4NV8BkmFhBzqjix7XL9DsKPrua7OaMc7t8QKGw_3Ut5wyv4NL4FHX74JFEEbmpVbsPINN7LcqLtewuyF0o0P9461PY9qLBAGy6Wr7PyE0qwDogQoDGJ1UJgPn"
    #
    self.fname_id = 0
    self.dname_img = "img_colab/"
    self.flops_per_sec_gcolab_cpu = 4887694725 # 925,554,209 | 9,276,182,810 | 1,722,089,747 | 5,287,694,725
    self.flops_per_sec_gcolab_gpu = 6365360673 # 1,021,721,764 | 9,748,048,188 | 2,245,406,502 | 6,965,360,673
    self.fname_requirements = './pluto_happy/requirements.txt'
    #
    self.color_primary = '#2780e3' #blue
    self.color_secondary = '#373a3c' #dark gray
    self.color_success = '#3fb618' #green
    self.color_info = '#9954bb' #purple
    self.color_warning = '#ff7518' #orange
    self.color_danger = '#ff0039' #red
    self.color_mid_gray = '#495057'
    self._xkeyfile = '.xoxo'
    return
  #
  # pretty print output name-value line
  def _pp(self, a, b,is_print=True):

    """
    Pretty print output name-value line

    Args:
        a (str) :
        b (str) :
        is_print (bool): whether to print the header or footer lines to console or return a str.

    Returns:
        y : None or output as (str)

    """
    # print("%34s : %s" % (str(a), str(b)))
    x = f'{"%34s" % str(a)} : {str(b)}'
    y = None
    if (is_print):
      print(x)
    else:
      y = x
    return y
  #
  # pretty print the header or footer lines
  def _ph(self,is_print=True):
    """
    Pretty prints the header or footer lines.

    Args:
      is_print (bool): whether to print the header or footer lines to console or return a str.

    Return:
      y : None or output as (str)

    """
    x = f'{"-"*34} : {"-"*34}'
    y = None
    if (is_print):
      print(x)
    else:
      y = x
    return y
  #
  # fetch huggingface file
  def fetch_hface_files(self,
    hf_names,
    hf_space="duchaba/monty",
    local_dir="/content/"):
    """
    Given a list of huggingface file names, download them from the provided huggingface space.

    Args:
        hf_names: (list) list of huggingface file names to download
        hf_space: (str) huggingface space to download from.
        local_dir: (str) local directory to store the files.

    Returns:
        status: (bool) True if download was successful, False otherwise.
    """
    status = True
    # f = str(hf_names) + " is not iteratable, type: " + str(type(hf_names))
    try:
      for f in hf_names:
        lo = local_dir + f
        huggingface_hub.hf_hub_download(repo_id=hf_space,
          filename=f,
          use_auth_token=True,
          repo_type=huggingface_hub.REPO_TYPE_SPACE,
          force_filename=lo)
    except:
      self._pp("*Error", f)
      status = False
    return status
  #
  # push files to huggingface
  def push_hface_files(self,
    hf_names,
    hf_space="duchaba/skin_cancer_diagnose",
    local_dir="/content/"):
    # push files to huggingface space

    """
    Pushes files to huggingface space.

    The function takes a list of file names as a
    paramater and pushes to the provided huggingface space.

    Args:
        hf_names: list(of strings), list of file names to be pushed.
        hf_space: (str), the huggingface space to push to.
        local_dir: (str), the local directory where the files
        are stored.

    Returns:
        status: (bool) True if successfully pushed else False.
    """
    status = True
    try:
      for f in hf_names:
        lo = local_dir + f
        huggingface_hub.upload_file(
          path_or_fileobj=lo,
          path_in_repo=f,
          repo_id=hf_space,
          repo_type=huggingface_hub.REPO_TYPE_SPACE)
    except Exception as e:
      self._pp("*Error", e)
      status = False
    return status
  #
  # push the folder to huggingface space
  def push_hface_folder(self,
    hf_folder,
    hf_space_id,
    hf_dest_folder=None):

    """

    This function pushes the folder to huggingface space.

    Args:
      hf_folder: (str). The path to the folder to push.
      hf_space_id: (str). The space id to push the folder to.
      hf_dest_folder: (str). The destination folder in the space. If not specified,
        the folder name will be used as the destination folder.

    Returns:
      status: (bool) True if the folder is pushed successfully, otherwise False.
    """

    status = True
    try:
      api = huggingface_hub.HfApi()
      api.upload_folder(folder_path=hf_folder,
        repo_id=hf_space_id,
        path_in_repo=hf_dest_folder,
        repo_type="space")
    except Exception as e:
      self._pp("*Error: ",e)
      status = False
    return status
  #
  # automatically restart huggingface space
  def restart_hface_periodically(self):

    """
    This function restarts the huggingface space automatically in random
    periodically.

    Args:
        None

    Returns:
        None
    """

    while True:
        random_time = random.randint(15800, 21600)
        time.sleep(random_time)
        os.execl(sys.executable, sys.executable, *sys.argv)
    return
  #
  # log into huggingface
  def login_hface(self, key=None):

    """
    Log into HuggingFace.

    Args:
      key: (str, optional)  If key is set, this key will be used to log in,
        otherwise the key will be decrypted from the key file.

    Returns:
        None
    """

    if (key is None):
      x = self._decrypt_it(self._huggingface_crkey)
    else:
      x = key
    huggingface_hub.login(x, add_to_git_credential=True) # non-blocking login
    self._ph()
    return
  #
  # Define a function to display available CPU and RAM
  def fetch_info_system(self):

    """
    Fetches system information, such as CPU usage and memory usage.

    Args:
        None.

    Returns:
        s: (str) A string containing the system information.
    """

    s=''
    # Get CPU usage as a percentage
    cpu_usage = psutil.cpu_percent()
    # Get available memory in bytes
    mem = psutil.virtual_memory()
    # Convert bytes to gigabytes
    mem_total_gb = mem.total / (1024 ** 3)
    mem_available_gb = mem.available / (1024 ** 3)
    mem_used_gb = mem.used / (1024 ** 3)
    # save the results
    s += f"Total memory: {mem_total_gb:.2f} GB\n"
    s += f"Available memory: {mem_available_gb:.2f} GB\n"
    # print(f"Used memory: {mem_used_gb:.2f} GB")
    s += f"Memory usage: {mem_used_gb/mem_total_gb:.2f}%\n"
    try:
      cpu_info = cpuinfo.get_cpu_info()
      s += f'CPU type: {cpu_info["brand_raw"]}, arch: {cpu_info["arch"]}\n'
      s += f'Number of CPU cores: {cpu_info["count"]}\n'
      s += f"CPU usage: {cpu_usage}%\n"
      s += f'Python version: {cpu_info["python_version"]}'
    except Exception as e:
      s += f'CPU type: Not accessible, Error: {e}'
    return s
  #
  # fetch GPU RAM info
  def fetch_info_gpu(self):

    """
    Function to fetch GPU RAM info

    Args:
        None.

    Returns:
        s: (str) GPU RAM info in human readable format.
    """

    s=''
    mtotal = 0
    mfree = 0
    try:
      nvml_handle = pynvml.nvmlInit()
      devices = pynvml.nvmlDeviceGetCount()
      for i in range(devices):
        device = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(device)
        mtotal += memory_info.total
        mfree += memory_info.free
      mtotal = mtotal / 1024**3
      mfree = mfree / 1024**3
      # print(f"GPU {i}: Total Memory: {memory_info.total/1024**3} GB, Free Memory: {memory_info.free/1024**3} GB")
      s += f'GPU type: {torch.cuda.get_device_name(0)}\n'
      s += f'GPU ready staus: {torch.cuda.is_available()}\n'
      s += f'Number of GPUs: {devices}\n'
      s += f'Total Memory: {mtotal:.2f} GB\n'
      s += f'Free Memory: {mfree:.2f} GB\n'
      s += f'GPU allocated RAM: {round(torch.cuda.memory_allocated(0)/1024**3,2)} GB\n'
      s += f'GPU reserved RAM {round(torch.cuda.memory_reserved(0)/1024**3,2)} GB\n'
    except Exception as e:
      s += f'**Warning, No GPU: {e}'
    return s
  #
  # fetch info about host ip
  def fetch_info_host_ip(self):
    """
    Function to fetch current host name and ip address

    Args:
        None.

    Returns:
        s: (str) host name and ip info in human readable format.
    """
    s=''
    try:
      hostname = socket.gethostname()
      ip_address = socket.gethostbyname(hostname)
      s += f"Hostname: {hostname}\n"
      s += f"IP Address: {ip_address}\n"
    except Exception as e:
      s += f"**Warning, No hostname: {e}"
    return s
  #
  # fetch files name
  def fetch_file_names(self,directory, file_extension=None):
    """
    This function gets all the filenames with a given extension.
    Args:
        directory (str):
            directory path to scan for files in.
        file_extension (list):
            file extension to look for or "None" (default) to get all files.
    Returns:
        filenames (list):
            list of strings containing the filenames with the given extension.
    """
    filenames = []
    for (root, subFolders, files) in os.walk(directory):
      for fname in files:
        if (file_extension is None):
          filenames.append(os.path.join(root, fname))
        else:
          for ext in file_extension:
            if fname.endswith(ext):
              filenames.append(os.path.join(root, fname))
    return filenames
  #
  # fetch the crypto key
  def _fetch_crypt(self,has_new_key=False):

    """
    This function fetches the crypto key from the file or from the
    variable created previously in the class.
    Args:
        has_new_key (bool):
            is_generate flag to indicate whether the key should be
            use as-is or fetch from the file.
    Returns:
        s (str):
            string value containing the crypto key.
    """
    if self._fkey == 'your_key_goes_here':
      raise Exception('Cryto Key is not correct!')
    #
    s=self._fkey[::-1]
    if (has_new_key):
      s=open(self._xkeyfile, "rb").read()
      self._fkey = s[::-1]
    return s
  #
  # generate new cryto key
  def gen_key(self):
    """
    This function generates a new cryto key and saves it to a file

    Args:
        None

    Returns:
        (str) crypto key
    """

    key = cryptography.fernet.Fernet.generate_key()
    with open(self._xkeyfile, "wb") as key_file:
        key_file.write(key[::-1]) # write in reversed
    return key
  #
  # decrypt message
  def decrypt_it(self, x):
    """
    Decrypts the encrypted string using the stored crypto key.

    Args:
        x: (str) to be decrypted.

    Returns:
        x: (str) decrypted version of x.
    """
    y = self._fetch_crypt()
    f = cryptography.fernet.Fernet(y)
    m = f.decrypt(x)
    return m.decode()
  #
  # encrypt message
  def encrypt_it(self, x):
    """
    encrypt message

    Args:
    x (str): message to encrypt

    Returns:
    str: encrypted message
    """

    key = self._fetch_crypt()
    p = x.encode()
    f = cryptography.fernet.Fernet(key)
    y = f.encrypt(p)
    return y
  #
  # fetch import libraries
  def _fetch_lib_import(self):

    """
    This function fetches all the imported libraries that are installed.

    Args:
        None

    Returns:
      x (list):
          list of strings containing the name of the imported libraries.
    """

    x = []
    for name, val in globals().items():
      if isinstance(val, types.ModuleType):
        x.append(val.__name__)
    x.sort()
    return x
  #
  # fetch lib version
  def _fetch_lib_version(self,lib_name):

    """
    This function fetches the version of the imported libraries.

    Args:
        lib_name (list):
            list of strings containing the name of the imported libraries.

    Returns:
        val (list):
            list of strings containing the version of the imported libraries.
    """

    val = []
    for x in lib_name:
      try:
        y = importlib.metadata.version(x)
        val.append(f'{x}=={y}')
      except Exception as e:
        val.append(f'|{x}==unknown_*or_system')
    val.sort()
    return val
  #
  # fetch the lib name and version
  def fetch_info_lib_import(self):
    """
    This function fetches all the imported libraries name and version that are installed.

    Args:
        None

    Returns:
      x (list):
          list of strings containing the name and version of the imported libraries.
    """
    x = self._fetch_lib_version(self._fetch_lib_import())
    return x
  #
  # write a file to local or cloud diskspace
  def write_file(self,fname, in_data):

    """
    Write a file to local or cloud diskspace or append to it if it already exists.

    Args:
        fname (str): The name of the file to write.
        in_data (list): The

    This is a utility function that writes a file to disk.
    The file name and text to write are passed in as arguments.
    The file is created, the text is written to it, and then the file is closed.

    Args:
        fname (str): The name of the file to write.
        in_data (list): The text to write to the file.

    Returns:
        None
    """

    if os.path.isfile(fname):
      f = open(fname, "a")
    else:
      f = open(fname, "w")
    f.writelines("\n".join(in_data))
    f.close()
    return
  #
  # fetch flops info
  def fetch_info_flops(self,model, input_shape=(1, 3, 224, 224), device="cpu", max_epoch=1):

    """
    Calculates the number of floating point operations (FLOPs).

    Args:
        model (torch.nn.Module): neural network model.
        input_shape (tuple): input tensor size.
        device (str): device to perform computation on.
        max_epoch (int): number of times

    Returns:
        (float): number of FLOPs, average from epoch, default is 1 epoch.
        (float): elapsed seconds
        (list): of string for a friendly human readable output
    """

    ttm_input = torch.rand(input_shape, dtype=torch.float32, device=device)
    # ttm_input = torch.rand((1, 3, 224, 224), dtype=torch.float32, device=device)
    tstart = time.time()
    for i in range(max_epoch):
      flops, params = flopth(model, inputs=(ttm_input,), bare_number=True)
    tend = time.time()
    etime = (tend - tstart)/max_epoch

    # kilo = 10^3, maga = 10^6, giga = 10^9, tera=10^12, peta=10^15, exa=10^18, zetta=10^21
    valstr = []
    valstr.append(f'Tensors device: {device}')
    valstr.append(f'flops: {flops:,}')
    valstr.append(f'params: {params:,}')
    valstr.append(f'epoch: {max_epoch}')
    valstr.append(f'sec: {etime}')
    # valstr += f'Tensors device: {device}, flops: {flops}, params: {params}, epoch: {max_epoch}, sec: {etime}\n'
    x = flops/etime
    y = (x/10**15)*86400
    valstr.append(f'Flops/s: {x:,}')
    valstr.append(f'PetaFlops/s: {x/10**15}')
    valstr.append(f'PetaFlops/day: {y}')
    valstr.append(f'1 PetaFlopsDay (on this system will take): {round(1/y, 2):,.2f} days')
    return flops, etime, valstr
  #
  def print_petaflops(self):

    """
    Prints the flops and peta-flops-day calculation. 
    **WARING**: This method will break/interfer with Stable Diffusion use of LoRA.
    I can't debug why yet.

    Args:
        None

    Returns:
        None    
    """
    self._pp('Model', 'TTM, Tiny Torch Model on: CPU')
    mtoy = TTM()
    # my_model = MyModel()
    dev = torch.device("cuda:0")
    a,b,c = self.fetch_info_flops(mtoy)
    y = round((a/b)/self.flops_per_sec_gcolab_cpu * 100, 2)
    self._pp('Flops', f'{a:,} flops')
    self._pp('Total elapse time', f'{b:,} seconds')
    self._pp('Flops compared', f'{y:,}% of Google Colab Pro')
    for i, val in enumerate(c):
      self._pp(f'Info {i}', val)
    self._ph()
    
    try:
      self._pp('Model', 'TTM, Tiny Torch Model on: GPU')
      dev = torch.device("cuda:0")
      a2,b2,c2 = self.fetch_info_flops(mtoy, device=dev)
      y2 = round((a2/b2)/self.flops_per_sec_gcolab_gpu * 100, 2)
      self._pp('Flops', f'{a2:,} flops')
      self._pp('Total elapse time', f'{b2:,} seconds')
      self._pp('Flops compared', f'{y2:,}% of Google Colab Pro')
      d2 = round(((a2/b2)/(a/b))*100, 2)
      self._pp('Flops GPU compared', f'{d2:,}% of CPU (or {round(d2-100,2):,}% faster)')
      for i, val in enumerate(c2):
        self._pp(f'Info {i}', val)
    except Exception as e:
      self._pp('Error', e)
    self._ph()    
    return
  #
  #
  def fetch_installed_libraries(self):
    """
    Retrieves and prints the names and versions of Python libraries installed by the user,
    excluding the standard libraries.

    Args:
    -----
      None

    Returns:
    --------
    dictionary: (dict)
      A dictionary where keys are the names of the libraries and values are their respective versions.

    Examples:
    ---------
      libraries = get_installed_libraries()
      for name, version in libraries.items():
        print(f"{name}: {version}")
    """
    # List of standard libraries (this may not be exhaustive and might need updates based on the Python version)
    # Run pip freeze command to get list of installed packages with their versions
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    
    # Decode result and split by lines
    packages = result.stdout.decode('utf-8').splitlines()

    # Split each line by '==' to separate package names and versions
    installed_libraries = {}
    for package in packages:
      try:
        name, version = package.split('==')
        installed_libraries[name] = version
      except Exception as e:
        #print(f'{package}: Error: {e}')
        pass
    return installed_libraries
  #
  #
  def fetch_match_file_dict(self, file_path, reference_dict):
    """
    Reads a file from the disk, creates an array with each line as an item,
    and checks if each line exists as a key in the provided dictionary. If it exists, 
    the associated value from the dictionary is also returned.

    Parameters:
    -----------
    file_path: str
        Path to the file to be read.
    reference_dict: dict
        Dictionary against which the file content (each line) will be checked.

    Returns:
    --------
    dict:
        A dictionary where keys are the lines from the file and values are either 
        the associated values from the reference dictionary or None if the key 
        doesn't exist in the dictionary.

    Raises:
    -------
    FileNotFoundError:
        If the provided file path does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Check if each line (stripped of whitespace and newline characters) exists in the reference dictionary.
    # If it exists, fetch its value. Otherwise, set the value to None.
    results = {line.strip(): reference_dict.get(line.strip().replace('_', '-'), None) for line in lines}

    return results
  # print fech_info about myself
  def print_info_self(self):

    """
    Prints information about the model/myself.

    Args:
        None

    Returns:
        None
    """

    self._ph()
    self._pp("Hello, I am", self.name)
    self._pp("I will display", "Python, Jupyter, and system info.")
    self._pp("For complete doc type", "help(pluto) ...or help(your_object_name)")
    self._pp('.','.')
    self._pp("...", "¯\_(ツ)_/¯")
    self._ph()
    # system
    self._pp('System', 'Info')
    x = self.fetch_info_system()
    print(x)
    self._ph()
    # gpu
    self._pp('GPU', 'Info')
    x = self.fetch_info_gpu()
    print(x)
    self._ph()
    # lib used
    self._pp('Installed lib from', self.fname_requirements)
    self._ph()
    x = self.fetch_match_file_dict(self.fname_requirements, self.fetch_installed_libraries())
    for item, value in x.items():
      self._pp(f'{item} version', value)
    self._ph()
    self._pp('Standard lib from', 'System')
    self._ph()
    self._pp('matplotlib version', matplotlib.__version__)
    self._pp('numpy version', numpy.__version__)
    self._pp('pandas version',pandas.__version__)
    self._pp('PIL version', PIL.__version__)
    self._pp('torch version', torch.__version__)
    self._ph()
    # host ip
    self._pp('Host', 'Info')
    x = self.fetch_info_host_ip()
    print(x)
    self._ph()
    #
    return
  #
# 
# define TTM for use in calculating flops
class TTM(torch.nn.Module):

  """
  Tiny Torch Model (TTM)

  This is a toy model consisting of four convolutional layers.

  Args:
      input_shape (tuple): input tensor size.

  Returns:
      (tensor): output of the model.
  """

  def __init__(self, input_shape=(1, 3, 224, 224)):
    super(TTM, self).__init__()
    self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    self.conv4 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

  def forward(self, x1):
    x1 = self.conv1(x1)
    x1 = self.conv2(x1)
    x1 = self.conv3(x1)
    x1 = self.conv4(x1)
    return x1
  #
# (end of class TTM)
# add module/method
#
import functools
def add_method(cls):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)
    setattr(cls, func.__name__, wrapper)
    return func # returning func means func can still be used normally
  return decorator
#
# [END OF pluto_happy]
#