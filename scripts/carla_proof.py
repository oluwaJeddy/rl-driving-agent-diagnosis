import carla, time, os
 
OUT = os.path.expanduser("~/rl-driving-agent-diagnosis/results/carla")
os.makedirs(OUT, exist_ok=True)
 
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world = client.get_world()
bp = world.get_blueprint_library()
 
# spawn a vehicle
vehicle_bp = bp.filter('vehicle.tesla.model3')[0]
spawn = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn)
vehicle.set_autopilot(True)
print("Vehicle spawned:", vehicle.id)
 
# attach a camera behind/above the car
cam_bp = bp.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
cam_tf = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
cam = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
 
saved = []
cam.listen(lambda img: (saved.append(img.frame),
                        img.save_to_disk(f"{OUT}/frame_{img.frame:06d}.png")))
 
print("Recording 5 seconds...")
time.sleep(5)
 
cam.stop()
cam.destroy()
vehicle.destroy()
print(f"Saved {len(saved)} frames to {OUT}")
