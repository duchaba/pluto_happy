
## required lib, required "pip install"
# import transformers
# import accelerate
import openai
import llama_index
import torch
import cryptography
import cryptography.fernet
## interface libs, required "pip install"
import gradio
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
# import PIL
# import pandas
import matplotlib
class HFace_Pluto(object):
  #
  # initialize the object
  def __init__(self, name="Pluto",*args, **kwargs):
    super(HFace_Pluto, self).__init__(*args, **kwargs)
    self.author = "Duc Haba"
    self.name = name
    self._ph()
    self._pp("Hello from class", str(self.__class__) + " Class: " + str(self.__class__.__name__))
    self._pp("Code name", self.name)
    self._pp("Author is", self.author)
    self._ph()
    #
    # define class var for stable division
    self._device = 'cuda'
    self._steps = [3,8,21,55,89,144]
    self._guidances = [1.1,3.0,5.0,8.0,13.0,21.0]
    self._xkeyfile = '.xoxo'
    self._models = []
    self._seed = 667 # sum of walnut in ascii (or Angle 667)
    self._width = 512
    self._height = 512
    self._step = 50
    self._guidances = 7.5
    self._llama_query_engine = None
    self._llama_index_doc = None
    self._llama_indexes_dict = None
    self._llama_query_engines_dict = None
    #self._generator = torch.Generator(device='cuda')
    self.pipes = []
    self.prompts = []
    self.images = []
    self.seeds = []
    self.fname_id = 0
    self.dname_img = "img_colab/"
    self._huggingface_key="gAAAAABkduT-XeiYtD41bzjLtwsLCe9y1FbHH6wZkOZwvLwCrgmOtNsFUPWVqMVG8MumazFhiUZy91mWEnLDLCFw3eKNWtOboIyON6yu4lctn6RCQ4Y9nJvx8wPyOnkzt7dm5OISgFcm"
    self._gpt_key="'gAAAAABkgiYGQY8ef5y192LpNgrAAZVCP3bo2za9iWSZzkyOJtc6wykLwGjFjxKFpsNryMgEhCATJSonslooNSBJFM3OcnVBz4jj_lyXPQABOCsOWqZm6W9nrZYTZkJ0uWAAGJV2B8uzQ13QZgI7VCZ12j8Q7WfrIg=='"
    self._fkey="=cvsOPRcWD6JONmdr4Sh6-PqF6nT1InYh965mI8f_sef"
    self._color_primary = '#2780e3' #blue
    self._color_secondary = '#373a3c' #dark gray
    self._color_success = '#3fb618' #green
    self._color_info = '#9954bb' #purple
    self._color_warning = '#ff7518' #orange
    self._color_danger = '#ff0039' #red
    self._color_mid_gray = '#495057'
    return
  #
  # pretty print output name-value line
  def _pp(self, a, b,is_print=True):
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
    f = str(hf_names) + " is not iteratable, type: " + str(type(hf_names))
    try:
      for f in hf_names:
        lo = local_dir + f
        huggingface_hub.hf_hub_download(repo_id=hf_space, filename=f,
          use_auth_token=True,repo_type=huggingface_hub.REPO_TYPE_SPACE,
          force_filename=lo)
    except:
      self._pp("*Error", f)
    return
  #
  #
  def push_hface_files(self,
    hf_names,
    hf_space="duchaba/skin_cancer_diagnose",
    local_dir="/content/"):
    f = str(hf_names) + " is not iteratable, type: " + str(type(hf_names))
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
    return
  #
  def push_hface_folder(self, hf_folder, hf_space_id, hf_dest_folder=None):
    api = huggingface_hub.HfApi()
    api.upload_folder(folder_path=hf_folder,
      repo_id=hf_space_id,
      path_in_repo=hf_dest_folder,
      repo_type="space")
    return
  #
  # Define a function to display available CPU and RAM
  def fetch_system_info(self):
    s=''
    # Get CPU usage as a percentage
    cpu_usage = psutil.cpu_percent()
    # Get available memory in bytes
    mem = psutil.virtual_memory()
    # Convert bytes to gigabytes
    mem_total_gb = mem.total / (1024 ** 3)
    mem_available_gb = mem.available / (1024 ** 3)
    mem_used_gb = mem.used / (1024 ** 3)
    # Print the results
    s += f"CPU usage: {cpu_usage}%\n"
    s += f"Total memory: {mem_total_gb:.2f} GB\n"
    s += f"Available memory: {mem_available_gb:.2f} GB\n"
    # print(f"Used memory: {mem_used_gb:.2f} GB")
    s += f"Memory usage: {mem_used_gb/mem_total_gb:.2f}%\n"
    return s
  #
  def restart_script_periodically(self):
    while True:
      #random_time = random.randint(540, 600)
      random_time = random.randint(15800, 21600)
      time.sleep(random_time)
      os.execl(sys.executable, sys.executable, *sys.argv)
    return
  #
  def write_file(self,fname, txt):
    f = open(fname, "w")
    f.writelines("\n".join(txt))
    f.close()
    return
  #
  def fetch_gpu_info(self):
    s=''
    try:
      s += f'Your GPU is the {torch.cuda.get_device_name(0)}\n'
      s += f'GPU ready staus {torch.cuda.is_available()}\n'
      s += f'GPU allocated RAM: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB\n'
      s += f'GPU reserved RAM {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB\n'
    except Exception as e:
      s += f'**Warning, No GPU: {e}'
    return s
  #
  def _fetch_crypt(self,is_generate=False):
    s=self._fkey[::-1]
    if (is_generate):
      s=open(self._xkeyfile, "rb").read()
    return s
  #
  def _gen_key(self):
    key = cryptography.fernet.Fernet.generate_key()
    with open(self._xkeyfile, "wb") as key_file:
        key_file.write(key)
    return
  #
  def _decrypt_it(self, x):
    y = self._fetch_crypt()
    f = cryptography.fernet.Fernet(y)
    m = f.decrypt(x)
    return m.decode()
  #
  def _encrypt_it(self, x):
    key = self._fetch_crypt()
    p = x.encode()
    f = cryptography.fernet.Fernet(key)
    y = f.encrypt(p)
    return y
  #
  def _login_hface(self):
    huggingface_hub.login(self._decrypt_it(self._huggingface_key),
      add_to_git_credential=True) # non-blocking login
    self._ph()
    return
  #
  def _fetch_version(self):
    s = ''
    # print(f"{'torch: 2.0.1':<25} Actual: {torch.__version__}")
    # print(f"{'transformers: 4.29.2':<25} Actual: {transformers.__version__}")
    s += f"{'openai: 0.27.7,':<28} Actual: {openai.__version__}\n"
    s += f"{'huggingface_hub: 0.14.1,':<28} Actual: {huggingface_hub.__version__}\n"
    s += f"{'gradio: 3.32.0,':<28} Actual: {gradio.__version__}\n"
    s += f"{'cryptography: 41.0.1,':<28} Actual: {cryptography.__version__}\n"
    s += f"{'llama_index: 0.6.21.post1,':<28} Actual: {llama_index.__version__}\n"
    return s
  #
  def _fetch_host_ip(self):
    s=''
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    s += f"Hostname: {hostname}\n"
    s += f"IP Address: {ip_address}\n"
    return s
  #
  def _setup_openai(self,key=None):
    if (key is None):
      key = self._decrypt_it(self._gpt_key)
    #
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key
    return
  #
  def _fetch_index_files(self,llama_ix):
    res = []
    x = llama_ix.ref_doc_info
    for val in x.values():
      jdata = json.loads(val.to_json())
      try:
        fname = jdata['extra_info']['file_name']
        res.append(fname)
      except:
        fname = jdata['metadata']['file_name']
        res.append(fname)
    # remove dublication name
    res = list(set(res))
    return res
  #
  def _fetch_dir_name(self,directory):
    dname=[]
    for name in os.listdir(directory):
      if os.path.isdir(os.path.join(directory, name)):
        if (name[0] != '.'):
          print(name)
          dname.append(name)
    dname.sort()
    return dname
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
