import numpy as np
import skrobot
from tendon import Tendon
import time


def beta_acute(alpha):
    return 2.0 * np.arctan(((3-2*np.sqrt(2)) * np.sin(alpha)) / (1-np.cos(alpha)))

def main():
    # Parallel links as cruciate ligaments
    robot_fk_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file="./urdf/two_link_fk.urdf") # Tibia
    robot_ik_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file="./urdf/two_link_ik.urdf") # Femur

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(robot_fk_model)
    viewer.add(robot_ik_model)

    ik_tip = skrobot.model.Box(extents=(0.01, 0.01, 0.01), face_colors=(0.75, 0.75, 0.75, 0.5))
    ik_tip.translate((0, 0, 0))
    robot_ik_model.ee_link.assoc(ik_tip, relative_coords='local')
    robots = [robot_fk_model, ik_tip]
    # robots = [ik_tip]


    ## Femur link attach
    femur_link = skrobot.model.MeshLink('./urdf/femur_sdf.stl', with_sdf=True)
    femur_link.set_color([0.0, 0.75, 0.0, 0.5])
    femur_link.translate((0.0, 0.015, 0.0))
    robot_ik_model.ee_link.assoc(femur_link, relative_coords='local')
    viewer.add(femur_link)

    ## Tibia link attach
    tibia_link = skrobot.model.MeshLink('./urdf/tibia_sdf.stl', with_sdf=True)
    tibia_link.set_color([0.0, 0.75, 0.0, 0.5])
    tibia_link.translate((0.0, 0.0, 0.016))
    robot_fk_model.tibia.assoc(tibia_link, relative_coords='local')
    viewer.add(tibia_link)


    ## Tibia ligament attachment points
    attachment_tibia_side = skrobot.model.Sphere(radius=(0.005), color=(0, 1, 0, 0.5))
    attachment_tibia_side.translate((0, 0.035, 0.015))
    robot_fk_model.tibia.assoc(attachment_tibia_side, relative_coords='local')

    attachment_tibia_forward = skrobot.model.Sphere(radius=(0.005), color=(0, 1, 0, 0.5))
    attachment_tibia_forward.translate((0.01, 0.02, -0.01))
    robot_fk_model.tibia.assoc(attachment_tibia_forward, relative_coords='local')

    attachment_tibia_backward = skrobot.model.Sphere(radius=(0.005), color=(0, 1, 0, 0.5))
    attachment_tibia_backward.translate((-0.01, 0.02, -0.01))
    robot_fk_model.tibia.assoc(attachment_tibia_backward, relative_coords='local')

    attachment_tibia_list = [attachment_tibia_side, attachment_tibia_forward, attachment_tibia_backward]
    for attachment_tibia in attachment_tibia_list:
        viewer.add(attachment_tibia)


    ## Femur ligament attachment points
    attachment_femur_list = []
    circle_points = 6
    for i in range(circle_points):
        radius = 0.015
        angle_offset = np.pi/2
        angle = -i * 2 * np.pi / circle_points + angle_offset
        attachment_femur = skrobot.model.Sphere(radius=(0.005), color=(0, 1, 0, 0.5))
        attachment_femur.translate((radius * np.cos(angle), 0.035, radius * np.sin(angle)))
        robot_ik_model.ee_link.assoc(attachment_femur, relative_coords='local')
        attachment_femur_list.append(attachment_femur)

    attachment_femur_distant = skrobot.model.Sphere(radius=(0.005), color=(0, 1, 0, 0.5))
    attachment_femur_distant.translate((0.0025, 0.035, 0.04))
    robot_ik_model.ee_link.assoc(attachment_femur_distant, relative_coords='local')
    attachment_femur_list.append(attachment_femur_distant)


    for attachment_femur in attachment_femur_list:
        viewer.add(attachment_femur)




    #joint update
    alpha = np.pi * 1 / 4.0
    beta = beta_acute(alpha)
    theta = -alpha + beta

    robot_fk_model.joint2.joint_angle(alpha - np.pi)
    robot_ik_model.joint1.joint_angle(-beta_acute(alpha))
    robot_ik_model.joint2.joint_angle(alpha - np.pi)

    #view axes for checking
    for robot in robots:
        axis = skrobot.model.Axis(
            axis_radius=0.003,
            axis_length=0.05,
            pos=robot.worldpos(),
            rot=robot.worldrot()
        )
        viewer.add(axis)

    viewer.show()


    # i_max = 100
    # n_femur = len(attachment_femur_list)
    # tibia_num = 2

    # # Header
    # header = ["i", "alpha", "beta", "theta"] + [f"T#{tibia_num}->F#{j}" for j in range(n_femur)]
    # print(",".join(header))

    # for i in range(i_max):
    #     alpha = (i_max - i) / i_max * np.pi / 4.0
    #     beta = beta_acute(alpha)
    #     theta = -alpha + beta

    #     # joint update
    #     robot_fk_model.joint2.joint_angle(alpha - np.pi)
    #     robot_ik_model.joint1.joint_angle(-beta_acute(alpha))
    #     robot_ik_model.joint2.joint_angle(alpha - np.pi)

    #     tibia_pos = attachment_tibia_list[tibia_num].worldpos()

    #     row = [i, alpha, beta, theta]

    #     for femur in attachment_femur_list:
    #         femur_pos = femur.worldpos()
    #         dist = np.linalg.norm(tibia_pos - femur_pos)
    #         row.append(dist)

    #     # print like csv
    #     print(",".join(map(str, row)))

    time.sleep(10)


if __name__ == '__main__':
    main()