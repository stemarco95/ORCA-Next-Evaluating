from time import sleep

from core.runtime import Runtime


SYSTEMS = {
    "scenario1": {
        "PerfectSelector": "configs/Scenario1/PerfectSelector.json",
        "ModelCycler": "configs/Scenario1/ModelCycler.json",
        "ModelCyclerOptimal": "configs/Scenario1/ModelCyclerOptimal.json",
        "Robust": "configs/Scenario1/Robust.json",
        "NonAdaptive": "configs/Scenario1/NonAdaptive.json", 
    },
    "scenario2": {
        "AdaptiveLearner": "configs/Scenario2/AdaptiveLearner.json",
        "OptimisingLearner": "configs/Scenario2/OptimisingLearner.json",
        "NonAdaptive": "configs/Scenario2/NonAdaptive.json",
        "Failing": "configs/Scenario2/Failing.json"
    },
    "scenario3": {
        "AdaptiveLearner": "configs/Scenario3/AdaptiveLearner.json",
        "OptimisingLearner": "configs/Scenario3/OptimisingLearner.json",
        "NonAdaptive": "configs/Scenario3/NonAdaptive.json"
    }
}

def scenario1():
    runtime = Runtime.from_config(SYSTEMS["scenario1"]["ModelCyclerOptimal"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario1"]["ModelCycler"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario1"]["Robust"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario1"]["NonAdaptive"])
    runtime.run()

    runtime = Runtime.from_config(SYSTEMS["scenario1"]["PerfectSelector"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

def scenario3():

    runtime = Runtime.from_config(SYSTEMS["scenario3"]["AdaptiveLearner"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario3"]["OptimisingLearner"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario3"]["NonAdaptive"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime


def scenario2():
    runtime = Runtime.from_config(SYSTEMS["scenario2"]["Failing"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario2"]["NonAdaptive"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario2"]["AdaptiveLearner"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime

    runtime = Runtime.from_config(SYSTEMS["scenario2"]["OptimisingLearner"])
    runtime.run()

    sleep(5)  # Short delay to ensure logs are written before starting the next runtime



    
if __name__ == "__main__":
    scenario1()
    scenario3()
    scenario2()





    
    
