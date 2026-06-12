import numpy as np
#import autograd.numpy as np
#from autograd import grad

import skrobot
import inspect
from tendon import Tendon

import time




def main():

    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file="./urdf/rleg.urdf")

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    #viewer = skrobot.viewers.PyrenderViewer()

    viewer.add(robot_model)
    viewer.show()

    knee_extensor_femur = skrobot.model.Box(extents=(0.05, 0.05, 0.05), face_colors=(0.75, 0.75, 0.75, 0.5))
    knee_extensor_femur.translate((0.05, 0, -0.15))
    robot_model.femur.assoc(knee_extensor_femur, relative_coords='local')

    knee_extensor_tibia = skrobot.model.Box(extents=(0.05, 0.05, 0.05), face_colors=(0.75, 0.75, 0.75, 0.5))
    knee_extensor_tibia.translate((0.05, 0, -0.05))
    robot_model.tibia.assoc(knee_extensor_tibia, relative_coords='local')

    knee_extensor_via_0 = skrobot.model.Box(extents=(0.025, 0.025, 0.025), face_colors=(0.0, 0.75, 0.75, 0.5))
    knee_extensor_via_0.translate((0.05, 0, -0.3))
    robot_model.femur.assoc(knee_extensor_via_0, relative_coords='local')

    knee_extensor_via_1 = skrobot.model.Box(extents=(0.025, 0.025, 0.025), face_colors=(0.0, 0.75, 0.75, 0.5))
    knee_extensor_via_1.translate((0.05, 0, 0.0))
    robot_model.tibia.assoc(knee_extensor_via_1, relative_coords='local')

    knee_cylinder = skrobot.model.Cylinder(radius=0.04, height=0.08, face_colors=(0.75, 0.75, 0.75, 0.5), with_sdf=True)
    knee_cylinder.translate((0.0, 0.0, 0.0))
    knee_cylinder.rotate(np.pi / 2.0, 'x')

    robot_model.tibia.assoc(knee_cylinder, relative_coords='local')


    # set initial posture knee bent
    robot_model.hip_roll.joint_angle(-0.8)
    robot_model.hip_pitch.joint_angle(-0.8)
    robot_model.hip_yaw.joint_angle(0.0)
    robot_model.knee_joint.joint_angle(1.2)


    knee_extensor_tendon = Tendon(name='knee_extensor', origin=knee_extensor_femur, insertion=knee_extensor_tibia, via=[knee_extensor_via_0, knee_extensor_via_1], 
                                  wrap=knee_cylinder, method='Newton')
    #knee_extensor_tendon = tendon('knee_extensor', knee_extensor_femur, knee_extensor_tibia, knee_extensor_via_0)

    print(knee_extensor_tendon.name)
    # print(knee_extensor_tendon.casadi_sdf)
    # print(knee_extensor_tendon.casadi_sdf())
    print(f'knee_extensor_femur is ... {knee_extensor_femur.worldpos()}')
    print(f'knee_extensor_tendon.origin.worldpos() is ... {knee_extensor_tendon.origin.worldpos()}')
    print(knee_extensor_tendon.origin.T())

    viewer.add(knee_extensor_femur)
    viewer.add(knee_extensor_tibia)
    viewer.add(knee_extensor_via_0)
    viewer.add(knee_extensor_via_1)
    viewer.add(knee_cylinder)
    viewer.add(knee_extensor_tendon.line_string)
    time.sleep(5)


    # set posture knee straight
    robot_model.hip_roll.joint_angle(0.0)
    robot_model.hip_pitch.joint_angle(0.0)
    robot_model.hip_yaw.joint_angle(0.0)

    robot_model.knee_joint.joint_angle(1.2)
    start_time = time.perf_counter()
    knee_extensor_tendon.redraw(viewer)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    time.sleep(5)

    #knee bend
    print(knee_extensor_tendon.name)
    # print(knee_extensor_tendon.casadi_sdf)
    # print(knee_extensor_tendon.casadi_sdf())
    robot_model.hip_roll.joint_angle(-0.8)
    robot_model.hip_pitch.joint_angle(-0.8)
    robot_model.hip_yaw.joint_angle(0.0)

    robot_model.knee_joint.joint_angle(2.6)

    start_time = time.perf_counter()
    knee_extensor_tendon.redraw(viewer)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    print(f'knee_extensor_femur.worldpos() is ... {knee_extensor_femur.worldpos()}')
    print(f'knee_extensor_tendon.origin.worldpos() is ... {knee_extensor_tendon.origin.worldpos()}')
    print(knee_extensor_tendon.origin.T())
    time.sleep(5)


if __name__ == '__main__':
    main()