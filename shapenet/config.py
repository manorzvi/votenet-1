class ShapenetConfig:
    def __init__(self):
        self.classname2originalcode = {
            'airplane' : '02691156',
            'bag' : '02773838',
            'cap' : '02954340',
            'car' : '02958343',
            'chair' : '03001627',
            'earphone' : '03261776',
            'guitar' : '03467517',
            'knife' : '03624134',
            'lamp' : '03636649',
            'laptop' : '03642806',
            'motorbike' : '03790512',
            'mug' : '03797390',
            'pistol' : '03948459',
            'rocket' : '04099429',
            'skateboard' : '04225987',
            'table' : '04379243'
        }
        self.originalcode2classname = {
            v : k for k, v in self.classname2originalcode.items()
        }
        self.classname2class = {
            c : i for i, c in enumerate(self.classname2originalcode.keys())
        }
        self.class2classname = {
            v : k for k, v in self.classname2class.items()
        }


if __name__ == '__main__':
    shapenet_config = ShapenetConfig()
    print(shapenet_config.__getattribute__('classname2class'))
    print(shapenet_config.__getattribute__('class2classname'))
