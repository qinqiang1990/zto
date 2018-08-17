# coding:utf-8

import configparser


def getConfig(section, key):
    config = configparser.ConfigParser()
    path = 'config/configure.ini'
    config.read(path)
    return config.get(section, key)


if __name__ == '__main__':
    epochs = getConfig("train", "epochs")
    print(epochs)
