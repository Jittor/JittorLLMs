import argparse

import models



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.availabel_models)
    
    args = parser.parse_args()
    
    model = models.get_model(args)
    print(model.run("张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，"))
