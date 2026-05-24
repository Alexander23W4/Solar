class AngleGenerator:
    def __init__(self, n, base_angle=0):
        """
            n : 面数
            base_angle : 基角度，起始角度
        """
        self.n = n
        self.base_angle = base_angle

    def generate(self):
        """
            List[float]: 角度列表
        """
        step = 360 / self.n
        return [(self.base_angle + i * step) % 360 for i in range(self.n)]


if __name__ == "__main__":
    gen1 = AngleGenerator(3, 13.4)
    print(gen1.generate())  # [0.0, 120.0, 240.0]

    gen2 = AngleGenerator(4, 60)
    print(gen2.generate())  # [60.0, 150.0, 240.0, 330.0]
