# [BEGIN OF pluto_happy]
# required pip install
import pynvml # for GPU info
## standard libs, no need to install
import numpy
import PIL
import pandas
import matplotlib
import torch
# standard libs (system)
import json
import time
import os
import random
import re
import sys
import psutil
import socket
import importlib.metadata
import types
import cpuinfo
import pathlib
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
 
  # Define a function to display available CPU and RAM
  def fetch_info_system(self, is_print=False):

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
    #
    # print it nicely
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
      if (is_print is True):
        self._ph()
        self._pp("System", "Info")
        self._ph()
        self._pp("Total Memory", f"{mem_total_gb:.2f} GB")
        self._pp("Available Memory", f"{mem_available_gb:.2f} GB")
        self._pp("Memory Usage", f"{mem_used_gb/mem_total_gb:.2f}%")
        self._pp("CPU Type", f'{cpu_info["brand_raw"]}, arch: {cpu_info["arch"]}')
        self._pp("CPU Cores Count", f'{cpu_info["count"]}')
        self._pp("CPU Usage", f"{cpu_usage}%")
        self._pp("Python Version", f'{cpu_info["python_version"]}')
    except Exception as e:
      s += f'CPU type: Not accessible, Error: {e}'
      if (is_print is True):
        self._ph()
        self._pp("CPU", f"*Warning* No CPU Access: {e}")
    return s
  #
  # fetch GPU RAM info
  def fetch_info_gpu(self, is_print=False):

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
      if (is_print is True):
        self._ph()
        self._pp("GPU", "Info")
        self._ph()
        self._pp("GPU Type", f'{torch.cuda.get_device_name(0)}')
        self._pp("GPU Ready Status", f'{torch.cuda.is_available()}')
        self._pp("GPU Count", f'{devices}')
        self._pp("GPU Total Memory", f'{mtotal:.2f} GB')
        self._pp("GPU Free Memory", f'{mfree:.2f} GB')
        self._pp("GPU allocated RAM", f'{round(torch.cuda.memory_allocated(0)/1024**3,2)} GB')
        self._pp("GPU reserved RAM", f'{round(torch.cuda.memory_reserved(0)/1024**3,2)} GB')
    except Exception as e:
      s += f'**Warning, No GPU: {e}'
      if (is_print is True):
        self._ph()
        self._pp("GPU", f"*Warning* No GPU: {e}")
    return s
  #
  # fetch info about host ip
  def fetch_info_host_ip(self, is_print=True):
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
      if (is_print is True):
        self._ph()
        self._pp('Host and Notebook', 'Info')
        self._ph()
        self._pp('Host Name', f"{hostname}")
        self._pp("IP Address", f"{ip_address}")
        try:
          from jupyter_server import serverapp 
          self._pp("Jupyter Server", f'{serverapp.__version__}')
        except ImportError:
          self._pp("Jupyter Server", "Not accessible")
        try:
          import notebook 
          self._pp("Jupyter Notebook", f'{notebook.__version__}')
        except ImportError:
          self._pp("Jupyter Notebook ", "Not accessible")
    except Exception as e:
      s += f"**Warning, No hostname: {e}"
      if (is_print is True):
        self._ph()
        self._pp('Host Name and Notebook', 'Not accessible')
    return s
  #
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
    self._pp("Note", "For doc type: help(pluto) ...or help(your_object_name)")
    self._pp("Let Rock and Roll", "¯\_(ツ)_/¯")
    # system
    x = self.fetch_info_system(is_print=True)
    # print(x)
    # self._ph()
    # gpu
    # self._pp('GPU', 'Info')
    x = self.fetch_info_gpu(is_print=True)
    # print(x)
    self._ph()
    # lib used
    self._pp('Installed lib from', self.fname_requirements)
    self._ph()
    x = self.fetch_match_file_dict(self.fname_requirements, self.fetch_installed_libraries())
    for item, value in x.items():
      self._pp(f'{item} version', value)
    #
    self._ph()
    self._pp('Standard lib from', 'System')
    self._ph()
    self._pp('matplotlib version', matplotlib.__version__)
    self._pp('numpy version', numpy.__version__)
    self._pp('pandas version',pandas.__version__)
    self._pp('PIL version', PIL.__version__)
    self._pp('torch version', torch.__version__)
    #
    self.print_ml_libraries()
    # host ip
    x = self.fetch_info_host_ip()
    # print(x)
    self._ph()
    #
    return
  #
  def print_ml_libraries(self):
    """
    Checks for the presence of Gradio, fastai, huggingface_hub, and transformers libraries.

    Prints a message indicating whether each library is found or not.
    If a library is not found, it prints an informative message specifying the missing library.
    """
    self._ph()
    self._pp("ML Lib", "Info")
    try:
      import fastai
      self._pp("fastai", f"{fastai.__version__}")
    except ImportError:
      self._pp("fastai", "*Warning* library not found.")
    #
    try:
      import transformers
      self._pp("transformers", f"{transformers.__version__}")
    except ImportError:
      self._pp("transformers", "*Warning* library not found.") 
    #
    try:
      import diffusers
      self._pp("diffusers", f"{diffusers.__version__}")
    except ImportError:
      self._pp("diffusers",  "*Warning* library not found.") 
    #
    try:
      import gradio
      self._pp("gradio", f"{gradio.__version__}")
    except ImportError:
      self._pp("Gradio", "*Warning* library not found.")

    try:
      import huggingface_hub
      self._pp("HuggingFace Hub", f"{huggingface_hub.__version__}")
    except ImportError:
      self._pp("huggingface_hub", "*Warning* library not found.")
    return
# 
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
hanna = Pluto_Happy('Hanna, the explorer and ranger.')
hanna.fname_requirements = 'requirements.txt'
hanna.print_info_self()
#
import gradio
def greet(name):
    return "Hello " + name + "!!"

demo = gradio.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
#