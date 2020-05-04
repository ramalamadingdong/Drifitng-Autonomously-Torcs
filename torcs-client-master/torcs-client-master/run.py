#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver

if __name__ == '__main__':
    main(MyDriver(None))
