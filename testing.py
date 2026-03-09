from framework.device import MyPhone

pixel = MyPhone()
pixel.connect_device()
print(pixel)
pixel.unlock_lockscreen([7, 5, 3, 8, 8, 0])

# pixel.swipe((500, 500), (450, 1500))
# device.shell('input tap 500 500')
# device.shell('input swipe 500 1800 500 1400')
