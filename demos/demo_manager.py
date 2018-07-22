import settings
import logging
import log
import os
import demos.demo01
import demos.demo02
import demos.demo03


def exec_demo(demo_id):
    demo_params = settings.get_demo_settings(demo_id)

    configure_logger(demo_params)

    if demo_id==1:
        demos.demo01.exec_demo(demo_params)
        pass
    elif demo_id == 2:
        demos.demo02.exec_demo(demo_params)
    elif demo_id == 3:
        demos.demo03.exec_demo(demo_params)

def configure_logger(demo_params):
    # initialize output folder
    output_folder = demo_params["output_folder"]
    os.system("rm -rf " + output_folder)
    os.makedirs(output_folder, exist_ok=True)
    log.log_fileHandler = logging.FileHandler(
        output_folder+"/"+demo_params["name"]+".log")

