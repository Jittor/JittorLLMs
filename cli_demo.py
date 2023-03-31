import argparse

import models



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.availabel_models)
    
    args = parser.parse_args()
    
    model = models.get_model(args)

    name = {
        'pangualpha': '盘古α',
        'chatglm': 'ChatGLM',
        'chatrwkv': 'ChatRWKV',
        'llama': 'LLaMa'
    }

    while True:
        text = input("用户输入:")
        print("{}：".format(name[args.model]), end='')
        output = model.run(text)
        print("{}".format(output))

    #print(model.run("张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，"))
