# of_common.py
import os
import jinja2 as jj
from smartsim import Experiment
from smartredis import Client

env = jj.Environment()

def connect():
    of_case_name = os.environ["CASED"]
    fn_name = "pEqnInnerTest"

    addr = os.environ["SSDB"]                 # "host:port"
    cluster = (os.environ.get("SSDB_CLUSTER","0") == "1")

    client = Client(address=addr, cluster=cluster)
    return client, of_case_name, fn_name

def get_field_name(client, fn_name, field_name, processor=0, timestep=None):
    client.poll_dataset(fn_name + "_metadata", 10, 1000)
    meta = client.get_dataset(fn_name + "_metadata")

    ds_naming = env.from_string(str(meta.get_meta_strings("dataset")[0]))
    ds_name = ds_naming.render(time_index=timestep, mpi_rank=processor)

    f_naming = env.from_string(str(meta.get_meta_strings("field")[0]))
    f_name = f_naming.render(name=field_name, patch="internal")

    return f"{{{ds_name}}}.{f_name}"

