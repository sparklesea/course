raw:
self.conv = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 32, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4),
        )
gai:
self.conv = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 64, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4),
        )
new:
self.conv = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, 2),
            nn.ReLU(),
            nn.Conv2d(10,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4),
        )