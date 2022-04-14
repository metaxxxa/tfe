#!/usr/bin/python3

class NotAValidOption(Exception):
    """
    Checking for a valid list option
    """

    def __init__(self, option, valid_options):
        self.option = option
        self.valid_options = valid_options

        self.message = self.option+" is not a valid option, "+str(self.valid_options)+" are the only valid options"

        super().__init__(self.message)

class CannotCreateModel(Exception):
    """
    Cannot Create a Model
    """

    def __init__(self, message):
        self.message = message

        super().__init__(self.message)