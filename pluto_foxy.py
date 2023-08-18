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
import pkg_resources
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
    standard_libs = list(sys.builtin_module_names)

    installed_libraries = {}
    
    # Iterate over all installed distributions
    for dist in pkg_resources.working_set:
      if dist.project_name.lower() not in standard_libs:
        installed_libraries[dist.project_name] = dist.version

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
    results = {line.strip(): reference_dict.get(line.strip(), None) for line in lines}

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
# [BEGIN OF pluto_foxy]
#
# prompt: use gradio to load model stabilityai/stable-diffusion-xl-base-1.0
# grade: F // Not even close to working code. It is because the "load()" function is
# introduced about a year ago (too recent?)
#
#
import gradio
import transformers
import torch
import diffusers

#
@add_method(Pluto_Happy)
def fetch_auto_load(self, model='stabilityai/stable-diffusion-xl-base-1.0'):
  """
  This function is used to load HuggingFace pretrained model and run inference.
  
  Args:
    model: A string param. The name of a pretrained model. 
    Default is "stabilityai/stable-diffusion-xl-base-1.0"

  Returns:
    None
  """

  model= f'models/{model}'
  title='Pluto: Latest Image Generation'
  desc='This space Pluto Sandbox.'
  examples=['Flowers in Spring', 'Bird in Summer', 'beautiful woman close up on face in autumn.', 'Old man close up on face in winter.']
  arti = f'Note: The underline model is: {model}'
  gradio.load(model,
    title=title,
    description=desc,
    examples=examples,
    article=arti).launch(debug=True)
  return

# prompt: write a function using StableDiffusionXLPipeline and huggingface stabilityai/stable-diffusion-xl-base-1.0 to display text to image with documentation
# grade: F // Nothing useable after 3 tries
#
# after I wrote the function, I asked it to write the documentation
#
# prompt: write python inline documentation for the following function: fetch_image_model
# grade: A- // it does not said I stored the pipe in self.pipe

import gradio
import transformers
import torch
import diffusers

@add_method(Pluto_Happy)
def fetch_image_model(self, model):

  """
  Description:

  This function is used to load a pre-trained Stable Diffusion model.

  Args:

    model (str):
      The name of the model to load.

  Returns:

    None (the pipe is safed in self.pipe)

  """

  self.device = 'cuda'
  pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")
  pipe.to(self.device)
  self.pipe = pipe
  self.model = model
  return

# prompt: write a function using torch.generator and StableDiffusionXLPipeline for image with documentation
# grade: C+ // tecnially it works with one error, but it is not what I am looking for.
# so I rewrite it.
#
# and I asked it to document my functin for me.
#
# prompt: write python inline documentation for the following function: draw_me
# grade: A // it writes good doc.

@add_method(Pluto_Happy)
def draw_me(self,
  prompt,
  negative_prompt,
  height,
  width,
  steps,
  seed,
  denoising_end,
  guidance_scale,
  prompt_2,
  negative_prompt_2
  ):

  """
  Generate image using the prompt using Stable Diffusion.

  Args:
    prompt (str): Prompt to generate image from. e.g.: "image of a cat."
    negative_prompt (str): Negative prompt to generate image from. Default: "incomplete".
    height (int): The height of the image to generate. Default: 768.
    width (int): The width of the image to generate. Default: 768.
    steps (int): Number of steps to run the diffusion model for. Default: 40.
    seed (int): Seed for the random number generator. Default: -1, any random seed

  Returns:
    PIL image.
  """

  # Initialize the diffusion model.
  # self.fetch_image_model(model=model)

  # Generate the image.
  gen = torch.Generator(device=self.device).manual_seed(seed)
  ximage = 1
  result = self.pipe(prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    height=height,
    width=width,
    denoising_end=denoising_end,
    guidance_scale=guidance_scale,
    prompt_2=prompt_2,
    negative_prompt_2=negative_prompt_2,
    num_images_per_prompt=ximage,
    generator=gen,
    output_type="pil",
    ).images
  torch.cuda.empty_cache()
  return result[0]

# prompt: write a function to define and launch the gradio interface with text for prompt and negative prompt and slider for steps, height, width, num image per prompt and a generator and output is an image
# grade: F // after a few tries with different prompt, nothing work. So I wrote it manually.
#
# prompt for doc
# prompt: write python inline documentation for the following function:
# grade: A // it writes good doc.

@add_method(Pluto_Happy)
def fetch_gradio_interface(self, predict_fn):

  """
  Description:

  This function is used to create a Gradio interface based on the `predict_fn` function.

  Args:

    predict_fn (function):
      The function that will be used to generate the image.

  Returns:

    gradio.Interface:
      The Gradio interface.

  """

  inp=[
    gradio.Textbox(label='Ask me what image do you want to draw.',
      value='A picture of a beautiful model on Hawaii beach with super realistic detail, in 4K clarity, soft background focus, and vibrant colors.'),
    gradio.Textbox(label='What do you do NOT want in the picture?', value='dirty, pornographic'),
    gradio.Slider(512, 1024, 768, step=128, label='Height'),
    gradio.Slider(512, 1024, 768, step=128, label='Width'),
    gradio.Slider(5, maximum=80, value=40, step=5, label='Number of Iterations'),
    gradio.Slider(minimum=1, step=1, maximum=1000000, randomize=True, label='Seed (Generate difference picture)'),
    gradio.Slider(0, maximum=1.0, value=1, step=0.02, label='Advance: denoising_end'),
    gradio.Slider(0.5, maximum=12.0, value=7.5, step=0.5, label='Advance: guidance_scale'),
    gradio.Textbox(label='Advance: prompt_2: for the second decoder.', value=''),
    gradio.Textbox(label='Advance: negative_prompt_2: for the second decoder.', value='pixel noise, , mishape feature')
    ]
  out=['image']
  title="Stable Diffusion XL model"
  desc='It is hacking time.'
  arti=f'This model is the {self.model}'
  inface = gradio.Interface(fn=predict_fn,
    inputs=inp,
    outputs=out,
    title=title,
    description=desc)
  return inface

# prompt: write the function from the above print dancer with documentation
# Note: 100% correct, but I did ask it write a function for printing a dancer is ascii art, but it could not do it.
# Note 2: I have to write the code with the comment "# print dancer" first.

@add_method(Pluto_Happy)
def dance_it(self):

  """
  This function prints a dancer

  Args:
    None

  Returns:
    None, just a print out
  """

  print('|-----------------------------------------------------------------------|')
  print('|    o   \ o /  _ o         __|    \ /     |__        o _  \ o /   o    |')
  print('|   /|\    |     /\   ___\o   \o    |    o/    o/__   /\     |    /|\   |')
  print('|   / \   / \   | \  /)  |    ( \  /o\  / )    |  (\  / |   / \   / \   |')
  print('|----------------------------Yahoo_ooo----------------------------------|')
  return
#

# prompt: define a function for print ascii art for the word happy
# Note: Failed. it could not do it. so I use https://patorjk.com with efti wall

@add_method(Pluto_Happy)
def print_monkey(self):
  """
  This function prints the ascii art for the word "happy".

  Args:
    None

  Returns:
    None
  """

  print("""
0----Monkey_See-------------.-----------------..----------------.--Monkey_Do---0
|                >     <    |                 ||                |    ._____.   |
0    ***         |.===.|    !=ooO=========Ooo=!!=ooO========Ooo=!    | -_- |   0
|   (o o)        {}o o{}       \\\\  (o o)  //      \\\\  (o o) //       ([o o])   |
ooO--(_)--Ooo-ooO--(_)--Ooo---------(_)----------------(_)--------ooO--(_)---Ooo
  """)
  return

# prompt: write new class Pluto_FastAI inherent from Pluto_Happy with documentation
# Note: 90% correct, the "init()" missing self and name parameter, and super() is wrong
# and I add in new method say_tagline() just for fun

class Pluto_FastAI(Pluto_Happy):
  """
  A class that inherits from Pluto_Happy, and add FastAI functionality

  Args:
      Pluto_Happy: A class that contains common functionality to Pluto.
  Returns:
      A class that contains both the functionality of Pluto_Happy and FastAI.
  """
  def __init__(self, name='Pluto',*args, **kwargs):
    super(Pluto_FastAI, self).__init__(name,*args, **kwargs)
    return
  #
  def say_tagline(self):
    """
    Print the tagline. For fun and no other purpose.

    Args:
      None.

    Returns:
      None
    """
    self._ph()
    self._pp('Call to arm:', 'I am Pluto the Seeker.')
    self._ph()
    return
# (end of Pluto_FastAI class)

# prompt: write documentation for the function fetch_image_url_online
# Grade: A // it can document good.
import duckduckgo_search
import fastai

# change name and imports to conform to Pluto standard
@add_method(Pluto_FastAI)
def fetch_image_url_online(self,term):

  """
  Searches for images of given term.

  Args:
    term: The term to search for.

  Returns:
    A list of dictionaries, each of which contains the following keys:
      title: The title of the image.
      image: The URL of the image.
      thumbnail: The URL of thumbnail of the image.
      url: The URL of the webpage containing the image.
      height: The height of the image in pixels.
      width: The width of the image in pixels.
      source: The source of the image.
  """

  d = duckduckgo_search.DDGS()
  val = d.images(term,size='Medium',type_image='photo',color='color')
  return val

# prompt: write a function to display an image from a URL with documentation
# Grade: B- // it works, but import is in function and not clean

# Import the necessary libraries.
import IPython
@add_method(Pluto_FastAI)
def draw_image_url(url):

  """
  Displays an image from a given URL.
  The image can be any format supported by PIL.
  The function uses the IPython.display library to display the image.

  Args:
    url: The URL from which to display the image.

  Returns:
    None
  """

  # Display the image.
  display(IPython.core.display.Image(url))
  return

# prompt: define a function to download image, save it in a directory and display it from url with error trapping and documentation
# Note: C- // I add imports, check for directory not exist,
# add default filename, and change the exception to print

# change name and conform to Pluto coding style
@add_method(Pluto_FastAI)
def _download_one_image(self,url, directory, filename, is_display=False):

  """
  Downloads an image from the given URL, saves it in the given directory, and displays it.

  Args:
    url: (str) The URL of the image to download.
    directory: (str) The directory to save the image in.
    filename: (str) The filename to save the image as.
    is_display: (bool) If True, display the image. Default is False

  Returns:
    None
  """

  try:
    # Download the image
    image_file = requests.get(url)

    # Create a directory if not exist
    if os.path.exists(directory) == False:
      os.makedirs(directory)

    # Save the image in the given directory
    with open(os.path.join(directory, filename), "wb") as f:
      f.write(image_file.content)
      f.close()

    # Display the image
    if is_display:
      print(f'{directory}/{filename}')
      img = PIL.Image.open(f'{directory}/{filename}')
      display(img)
  except Exception as e:
    print(f'Error: Can not download or display image: {directory}/{filename}.\nError: {e}')
  return

# prompt: write a function call fetch_images that combine _download_one_image and download_images with documentation
# Grade: B // It works, but I change filename format and add in parameter upto_max

# Upate to Pluto coding standard and name
# Fetch images
@add_method(Pluto_FastAI)
def download_images_from_search(self, term, directory, is_display=False, upto_max=300):

  """
  Searches for images of given term, downloads them, and saves them in the given directory.

  Args:
    term: (str) The term to search for.
    directory: (str) The directory to save the images in.
    is_display: (bool) If True, display the images. Default is False.
    upto_max: (int) The upto maximum number of images to download. Default is 300

  Returns:
    A list of dictionaries, each of which contains the following keys:

      title: The title of the image.
      image: The URL of the image.
      thumbnail: The URL of thumbnail of the image.
      url: The URL of the webpage containing the image.
      height: The height of the image in pixels.
      width: The width of the image in pixels.
      source: The source of the image.
    
    A list of images download file name
  """

  # Search for images
  images_info = self.fetch_image_url_online(term)

  # Download images
  id = 0
  img_download = []
  for ix in images_info:
    url = ix['image']
    # I add the clean filename below
    filename = f"{term.replace(' ','_')}-{id}.{url.rsplit('.', 1)[-1]}"
    res = re.split('[\\?\\!\\&]', filename)
    #
    filename = res[0]
    self._download_one_image(url, directory, filename, is_display)
    img_download.append(f'{directory}/{filename}')
    if id == upto_max:
      break
    id += 1

  # Display number of images download
  # print(f'Number of images download is: {id}')
  return images_info, img_download

# prompt: write a function to display thumb images from a directory of images in a row and column format
# Grade: C+ // The calculate of the indexes "ax" is wrong. I correct it. And it import numpy but not usig it.
# Note 2: it could be not an image so add in try: except:

import pathlib
# display thumb images
@add_method(Pluto_FastAI)
def draw_thumb_images(self,dname, nrows=2, ncols=4):

  """
  Displays thumb images from a directory or a Pandas dataframe of images in a row and column format.

  Args:
    directory: (str or DataFrame) The directory containing the images Or the dataframe.
    nrows: (int) The number of rows to display the images in. Default is 2 rows.
    ncols: (int) The number of columns to display the images in. Defaut is 4 columns.

  Returns:
    A list (list) of displayed images
  """

  # os.path.exists(directory)
  if isinstance(dname, str):
    # Get the list of images in the directory
    images = self.fetch_file_names(dname)
  else:
    # it got to be pandas dataframe
    images = dname.sample(nrows*ncols)

  # Create a figure with the specified number of rows and columns
  fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols)

  # keep track of img names
  img_names = []
  # Display the images in the figure
  for i, image in enumerate(images):
    if (i == (nrows * ncols)):
      break
    ax = axes[i // ncols, i % ncols]
    try:
      ax.imshow(matplotlib.pyplot.imread(image))
      ax.axis('off')
      img_names.append(image)
    except Exception as e:
      print(f'Error: Can not display image: {image}. Error: {e}')

  # Display the figure
  matplotlib.pyplot.tight_layout(pad=0.25)
  matplotlib.pyplot.show()
  return img_names

# prompt: write a new foxy function with documentation and error checking for the following: delete file with file extension not on a list, the file is in a directory
# Grade: A // it works, and I am getting smarter on how to phrase the prompt.

@add_method(Pluto_FastAI)
def delete_file_extensions(self,directory,file_ext_list):

  """
  Deletes files in a directory that are not in the file extension list.

  Args:
    directory: (str) The directory containing the files.
    file_ext_list: (list) The list of file extensions to keep. e.g. (".jpg", ".png")

  Returns:
    None:

  """

  # Get the list of files in the directory
  files = self.fetch_file_names(directory)
  file_delete = []

  # Delete files not in the extension list
  for file in files:
    file_ext = pathlib.Path(file).suffix
    if file_ext not in file_ext_list:
      os.remove(file)
      print(f'Deleting file not in extension list: {file}')
      file_delete.append(file)

  # Display a message indicating the completion of the operation
  # print(f'Deleting files not in extension list in {directory} is done!')
  return file_delete

# prompt: write a function for reading images from a directory if not an image then delete it
# Grade: A // It works, but it should close image before delete else it would be a race condition.

@add_method(Pluto_FastAI)
# delete non images file
def delete_non_image_files(self,directory):

  """
  Deletes non-image files from a directory.

  Args:
    directory: The directory to delete non-image files from.

  Returns:
    A list (list) of deleted image file name.
    A list (list) of deleted file not with image exention.
  """

  # Get the list of files in the directory
  img_types = ['.png', '.jpg', '.jpeg', '.gif']
  file_delete = self.delete_file_extensions(directory, img_types)
  files = self.fetch_file_names(directory)

  #check on how many files deleted
  total_deleted = 0
  img_delete = []

  # Delete non-image files
  for file in files:
    try:
      img = PIL.Image.open(file)
      img.draft(img.mode, (32,32))
      img.load()
      if not (img.mode == 'RGB'):
        img.close()
        os.remove(file)
        print(f'Delete image not color: {file}')
        total_deleted += 1
    except Exception as e:
      os.remove(file)
      print(f'Delete not image: {file}. Error: {e}')
      total_deleted += 1
      img_delete.append(file)

  # Display the number of files deleted
  print(f'Total deleted: {total_deleted}. Total available imges: {len(files)-total_deleted}')
  return img_delete, file_delete

# prompt: write a function to create a pandas dataframe with two columns from directory of files, the first column is the full path and the second is the name of the file.
# Grade: B // it works, but with some minor error, and I refactor the method because it is too messy.

import pandas
import pathlib

# update to Pluto standard naming convention
@add_method(Pluto_FastAI)
def bake_df_img_name(self, directory,label_fn=None):
  """
  Creates/Bakes a pandas dataframe with two columns from directory of files,
  the first column name is: "full_path"
  and the second name is: "label". It is the filename without the index number and extension.

  Args:
    directory: (str) The directory containing the files.
    label_fn: (funcion) Optional the function to define the label to be used.
    The defaul funtion strip all but the core file name.

  Returns:
    A pandas dataframe with two columns: "full_path" and "label".
  """

  # Get the list of files in the directory
  files = self.fetch_file_names(directory)

  # Create a pandas dataframe with two columns
  df = pandas.DataFrame(files, columns=['full_path'])

  # Add a column for the label field
  if label_fn is None:
    df['label'] = df['full_path'].apply(lambda x: re.split('[-]', str(pathlib.Path(x).name))[0])
  else:
    df['label'] = df['full_path'].apply(label_fn)

  # Return the dataframe
  return df

# prompt: write a function with documentation for the following: resize all images to a square, image in a directory, use fastai lib
# Grade: A- // it got it right using PIL but not fastai lib, and not set the size as parameter.
# Note: this time it got the @add_method correctly. Yahhoooo :-)

@add_method(Pluto_FastAI)
def resize_img_square(self, directory, img_size=512):

  """
  Resizes all images in a directory to a square.

  Args:
    directory: (str) The directory containing the images.
    img_size: (int) the square image size. Default is 512.

  Returns:
    A list (list) of image file that can not be resize:

  """

  img_error = []
  # Get the list of files in the directory
  files = self.fetch_file_names(directory)

  # Resize all images to a square
  for file in files:
    try:
      img = PIL.Image.open(file)  # I fixed this with PIL.
      img = img.resize((img_size, img_size))  # I fixed this.
      img.save(file)
    except Exception as e:
      print(f'Error file: {file}')
      print(f'Error: {e}')
      img_error.append(file)

  # Display a message indicating the completion of the resize operation
  # print(f'Resizing images in {directory} to square is done!')
  return img_error

# prompt: write a foxy function to download dataset from Kaggle website using opendatasets lib with documentation
# Grade: B- // It works, but it failded at first many tried. So, I told it "opendatasets" lib.

import opendatasets
import os

# Function to download dataset from Kaggle website using opendatasets lib.
@add_method(Pluto_FastAI)
def download_kaggle_dataset(self,dataset_name, path_to_save):

  """
  Downloads a dataset from Kaggle website using opendatasets library.

  Args:
    dataset_name: (str) The name of the dataset to download.
    path_to_save: (str) The path where the dataset will be saved.

  Returns:
    None
  """

  try:
    # Check if the dataset already exists
    if os.path.exists(path_to_save):
      print(f'Dataset {dataset_name} already exists.')
      return

    # Download the dataset
    print(f'Downloading dataset {dataset_name}...')
    opendatasets.download(dataset_name, path_to_save)
    print(f'Dataset {dataset_name} downloaded successfully.')

  except Exception as e:
    print(f'Error downloading dataset {dataset_name}: {e}')
  return None

# prompt: update function draw_diagram() with the following: change the node font to san serif
# prompt: 8 more updates prompts. (see #scratch Fun graph divergent section)
# Grade: B // after two hours of fun divergent, I got this to work

import graphviz

@add_method(Pluto_FastAI)
def draw_diagram(self, nodes, edges, labels, node_color=None, 
  horizontal=False, title='GraphViz', fontsize='10'):

  """Draws a diagram using Graphviz.

  Args:
    nodes: (list) A list of nodes.
    edges: (list) A list of edges.
    labels: (list) A list of labels for the nodes.
    node_color: (list) A list of colors for the nodes.
    horizontal: (bool) A boolean value indicating whether to display the diagram
      horizontally.
    fontsize: (str) The font size in point. Default is "10"

  Returns:
    A graph representation of the diagram.
  
  Example:
    nodes = ["A", "B", "C", "D", "E", "F"]
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), 
      ("F", "A"), ("D", "B"), ("E", "C")]
    labels = ["Node A", "Node B", "Node C", "Node D", "Node E", "Node F"]
    mute_colors = [
        "#e1a06c",
        "#c3ced1",
        "#e6dfda",
        "#c29d9e",
        "#df829d",
        "#e1a06c",
        "#c3ced1",
        "#e6dfda",
        "#c29d9e",
        "#df829d"
    ]
    # draw it
    diagram = draw_diagram(nodes, edges, labels, mute_colors, horizontal=True, title='Pluto Path to Success')

    # display it
    display(diagram)
  """

  mute_colors = [
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d",
    "#e1a06c",
    "#c3ced1",
    "#e6dfda",
    "#c29d9e",
    "#df829d"
  ]
  if node_color is None:
    node_color = mute_colors

  # Create a graph object.
  graph = graphviz.Digraph()

  # Add the nodes.
  for i, node in enumerate(nodes):
    graph.node(node, label=labels[i], color=node_color[i], 
      fontname='sans-serif', style='filled', fontsize=fontsize)

  # Add the edges.

  for edge in edges:
    graph.edge(edge[0], edge[1])

  # Set the title.
  graph.attr('graph', label=title)

  if horizontal:
    graph.attr('graph', rankdir='LR')
  else:
    graph.attr('graph', rankdir='TB')

  # Return the string representation of the diagram.

  return graph

# prompt: None
# Note: I am unsure how to describe the following function

# draw GraphViz for FastAI data block
@add_method(Pluto_FastAI)
def draw_fastai_data_block(self):
  nodes = ["A", "A1", "A2", "A3", "A4", "A5", "A6", 
    "B", "B1", "B2", "B3", "B4",
    "C", "C1", "C2", "C3", "D"]
  labels = ["@1_SOURCE", "Pandas", "X:Y Path", "X: Fn", "X:Y Block Type", "DataSet", "DataBlock",
    "@2_SPLIT", "80/20 Default", "Fn Custom", "DataSet", "DataBlock",
    "@3_TRANSFORM", "X: Transform", "X: Batch Augment", "OUT: DataLoader",
    "@4_DANCE :-)"]
  edges = [("A", "A6"), ("A6", "B"), ("B", "B4"), ("B4", "C"), ("C", "D"),
    ("A", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "A4"), ("A4", "A5"), 
    ("B", "B1"), ("B", "B2"), ("B1", "B3"),
    ("C", "C1"), ("C1", "C2"), ("C2", "C3")]
  #
  # draw it
  diagram = self.draw_diagram(nodes, edges, labels, node_color=None, 
    horizontal=True, title='Pluto view of FastAI Datablocks 3-Steps Plus Disco Dancing :-)',
    fontsize='8')

  # display it
  display(diagram)
  return diagram

# prompt: None
# Note: rewrite to be a function for foxy

@add_method(Pluto_FastAI)
def bake_dloader_from_file(self, directory, y_fn):
  dblock = fastai.data.block.DataBlock(
    get_items=fastai.data.transforms.get_image_files,
    get_y=y_fn)
  dset = dblock.datasets(directory)
  return dset, dblock

# prompt: write documentation for function foxy.bake_dloader_from_file()
# Grade: B // it does it correctly, except it return a datasets and not dataloader,
# and missing the add method

# I rewrote it for extentable
@add_method(Pluto_FastAI)
def bake_image_dblock_from_file(self, directory, y_fn, is_dataset=False, is_verbose=False):

  """
  Create a fastai datablock object from a directory of images.

  Args:
    directory: (str) A string path to the directory of images.
    y_fn: (fn) A function that takes a file path as input and returns the
      corresponding label.
    is_dataset: (bool) if True return a dataset or None. Default is False.
    is_verbose: (bool) print out step by step operation. Default is False.

  Returns:
    A fastai datablock object and datasets object.
  """

  dblock = fastai.data.block.DataBlock(
    get_items=fastai.data.transforms.get_image_files,
    get_y=y_fn,
    blocks = (fastai.vision.data.ImageBlock, fastai.vision.data.CategoryBlock))
  #
  dset = None
  if (is_dataset):
    dset = dblock.datasets(directory)
  if (is_verbose):
    try:
      dblock.summary(directory)
    except Exception as e:
      print(f'\n*Almost complete. Stop at: {e}')
  return dset, dblock

# prompt: No prompt
# Note: write from reading above code. I tried but failed to ask it to 
# write a function based on the above 3 code cells.

# show the pandas dataframe and display the y_label pie chart
@add_method(Pluto_FastAI)
def draw_df_ylabel(self, df,y_label='label'):
  df[y_label].value_counts().plot(kind='pie')
  display(df.describe())
  return

# prompt: None
# Note: I am unsure how to write the prompt for the following, other ask it to write document
# Document doc:
# prompt: write python detail inline documentation for the following function: bake_step1_data_source
# Grade: B // most of it correct

import fastai
from fastai.data.all import *
from fastai.vision.all import *
import torchvision
@add_method(Pluto_FastAI)
def bake_step1_data_source(self, df, x_col_index=0, y_col_index=1,is_verbose=False):

  """
  Create a fastai DataBlock and DataSet objects from a Pandas dataframe.
  The input (X) is the image full path.
  The label (Y) is the target

  Args:
    df: (pandas DataFrame) a dataframe of images with label.
    x_col_index: (int) index of the column that contains the image uri.
    y_col_index: (int) index of the column that contains the label.
    is_verbose: (bool) print out step by step operation. Default is False.

  Returns:
    A fastai datablock (DataBlock) object and datasets (DataSet) object.
  """  

  # step 1: Continue using Pandas
  dblock = fastai.data.block.DataBlock(
    get_x = fastai.data.transforms.ColReader(x_col_index),
    get_y = fastai.data.transforms.ColReader(y_col_index),
    blocks = (fastai.vision.data.ImageBlock, fastai.vision.data.CategoryBlock)
    )
  #
  dset = dblock.datasets(df)
  #
  if (is_verbose):
    self._ph()
    self._pp('Step 1 of 3', 'Source DataSet from Pandas')
    self._ph()
    print(f'Train: {dset.train[0]}, \nValid: {dset.valid[0]}')
    print(f'Vocab: {dset.vocab}, where 0 and 1 used as index')
    print(f'It does the auto split to train and valid. ')
    print(f'Size valid: {len(dset.valid)}')
    print(f'Total size: {len(dset.train)+len(dset.valid)}')
    print(f'Default spliter: 80/20: {str(dblock.splitter)}')
    # print out status
    self._ph()
    try:
      dblock.summary(df)
    except Exception as e:
      print(f'\n\n**Not yet complete. We stop at:\n{e}')
    self._ph()
    x = dset.train[0][0]
    display(x.show())
  return dset, dblock

# prompt: None
# Note: I am unsure how to write the prompt for the following, other ask it to write document
# use genAI to write doc.
# prompt: write python inline documentation for the following function: foxy.bake_step2_split
# grade: A // it know how to write doc.

@add_method(Pluto_FastAI)
def bake_step2_split(self, df, dblock, fn=None, is_verbose=False):

  """
  Split the DataFrame into training and validation datasets.

  Args:
    df: (pandas DataFrame) a dataframe of images with label.
    dblock: (fastai DataBlock) the datablock object.
    fn: (function) the spliter function. default is the default auto 80/20 split.
    is_verbose: (bool) print out step by step operation. Default is False.

  Returns:
    A fastai datablock (DataBlock) object and datasets (DataSet) object.
  """   
  if (fn is not None):
    dblock.splitter = fn
  #
  dset = dblock.datasets(df)
  #
  #
  if (is_verbose):
    self._ph()
    self._pp('Step 2 of 3', 'Split X (train) and Y (valid)')
    self._ph()
    print(f'Train: {dset.train[0]}, \nValid: {dset.valid[0]}')
    print(f'Vocab: {dset.vocab}, where 0 and 1 used as index')
    print(f'It does the auto split to train and valid. ')
    print(f'Size valid: {len(dset.valid)}')
    print(f'Total size: {len(dset.train)+len(dset.valid)}')
    print(f'Spliter: {str(dblock.splitter)}')
    # print out status
    self._ph()
    try:
      dblock.summary(df)
    except Exception as e:
      print(f'\n\n**Not yet complete. We stop at:\n{e}')
    self._ph()
    x = dset.train[0][0]
    display(x.show())
  return dset, dblock

# prompt: None
# Note: I am unsure how to write the prompt for the following, other ask it to write document

@add_method(Pluto_FastAI)
def bake_step3_transform(self, df, dblock, item_fn=None, batch_fn=None, is_verbose=False):

  """
  Transform the data into a DataSet and DataLoader objects.

  Args:
    df: (pandas DataFrame) a dataframe of images with label.
    dblock: (fastai DataBlock) the datablock object.
    item_fn: (function) the item transformer function. default is resize to 224.
    batch_fn: (function) the batch transformer function. default is default augmentation.
    is_verbose: (bool) print out step by step operation. Default is False.

  Returns:
    A fastai dataloader (DataLoader) object and datasets (DataSet) object.
  """    
  if (item_fn is None):
    dblock.default_item_tfms = fastai.vision.augment.Resize(224)
  else:
    dblock.default_item_tfms = item_fn
  #
  if (batch_fn is None):
    dblock.default_batch_tfms = fastai.vision.augment.aug_transforms() # use all the default settings
  else:
    dblock.default_batch_tfms = batch_fn

  dloader = dblock.dataloaders(df)
  #
  #
  if (is_verbose):
    self._ph()
    self._pp('Step 3 of 3', 'Item transform (resize), Batch transform (augmentation)')
    self._ph()
    print(f'Train: {dloader.train_ds[0]}, \nValid: {dloader.valid_ds[0]}')
    print(f'Vocab: {dloader.vocab}, where 0 and 1 used as index')
    print(f'Size valid: {len(dloader.valid_ds)}')
    print(f'Total size: {len(dloader.train_ds)+len(dloader.valid_ds)}')
    self._ph()
    print(f'Spliter: {str(dblock.splitter)}')
    self._ph()
    print(f'Item Transform: {str(dblock.default_item_tfms)}')
    self._ph()
    print(f'Batch Transform: {str(dblock.default_batch_tfms)}')
    # print out status
    self._ph()
    try:
      dblock.summary(df)
    except Exception as e:
      print(f'\n\n**Not yet complete. We stop at:\n{e}')
    self._ph()
    display(dloader.show_batch())
  return dloader, dblock

# prompt: None
# Note: I am unsure how to describe the following function

# draw GraphViz for FastAI data block
@add_method(Pluto_FastAI)
def draw_fastai_train(self):
  nodes = ["A", "A1", "A2", "A3", "A4",
    "B", "B1", "B2", 
    "C", "C1", "C2",
    "D"]
  labels = ["@1_LEARNER", "DataLoader", "Model Arch", "Error Metric", "Learning Rate", 
    "@2_FINE_TUNE", "Epoch", "Callback",
    "@3_MONITOR", "OUT: Save Model", "Break",
    "@4_TEA_BREAK :-)"]
  edges = [("A", "B"), ("C", "D"),
    ("A", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "A4"), 
    ("B", "B1"), ("B", "B2"), ("B2", "C"), 
    ("C", "C1"), ("C", "C2")]
  #
  # draw it
  diagram = self.draw_diagram(nodes, edges, labels, node_color=None, 
    horizontal=True, title='Pluto view of FastAI Learn Plus Disco Dancing :-)',
    fontsize='8')

  # display it
  display(diagram)
  return diagram

# prompt: write a function with documentation for the following: print all the name begin with partial label, variable avail_pretrained_models
# grade: A // it works

import timm
@add_method(Pluto_FastAI)
def fetch_timm_models_name(partial_label):

  """Return all the models name from timm library that begin with partial_label

  Args:
    partial_label (str): partial label for the model name

  Returns:
    A list of strings with the models name
  """

  avail_pretrained_models = timm.list_models(pretrained=True)
  models = [model for model in avail_pretrained_models if partial_label in model]
  #
  print(f'Total available models: {len(avail_pretrained_models)}')
  print(f'Total models with partial label {partial_label}: {len(models)} ')
  return models

# prompt: (from gpt4)
from fastai.callback.core import Callback
#
class StopAndSaveOnLowError(Callback):
  def __init__(self, threshold=0.009, fname='best_low_error_model'):
    self.threshold = threshold
    self.fname = fname
    return

  def after_epoch(self):
    # Assuming error_rate is a monitored metric
    if 'error_rate' in self.learn.recorder.metric_names:
      error = self.learn.recorder.log[self.learn.recorder.metric_names.index('error_rate')]
      if error <= self.threshold:
        self.fname = f'{self.fname}_{error:.4}'
        self.fname = self.fname.replace('.', 'd')
        self.learn.save(self.fname)
        print(f"Saving model as error rate {error} is less than {self.threshold}: Model name: {self.fname}")
        print(f"Stopping training as error rate {error} is less than {self.threshold}")
        raise CancelTrainException
    return
#
# END OF pluto_foxy
#