class DemoService:
    def __init__(self):
        self.hello_string = "Hellooooooo "
        self.goodby_string = "Goodbyeeee "

    def say_hello(self, nameOfUser)-> str:
        return self.hello_string + nameOfUser

    def say_goodbye(self, nameOfUser)-> str:
        return self.goodby_string + nameOfUser