import random
import math
import copy
import sys
import pygame
import timeit
import numpy as np
import matplotlib.pyplot as plt
import panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3
from panda3d.core import LVector3
from panda3d.core import PerspectiveLens
from panda3d.core import AmbientLight, DirectionalLight
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.MetaInterval import Sequence
from direct.interval.LerpInterval import LerpFunc
from direct.interval.FunctionInterval import Func
from direct.interval.ActorInterval import ActorInterval

show_animation = True

X_LIMIT = 1000
Y_LIMIT = 1000
Z_LIMIT = 1000


# windowSize = [X_LIMIT, Y_LIMIT, Z_LIMIT]

# pygame.init()
# fpsClock = pygame.time.Clock()

# screen = pygame.display.set_mode(windowSize)
# screen.fill((255, 255, 255))
# pygame.display.set_caption("RRT*")


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parentIndex = None
        self.isLeaf = True
        self.cost = 0.0


class RRTStar(ShowBase):
    def __init__(self, start, goal, obstacles, stepSize=5.0, iteration=500, goalRadius=10):
        """ RRT* initiation
        :param start: start position
        :param goal: goal position
        :param obstacles: obstacle list
        :param stepSize: process step
        :param iteration: loop limit
        """
        ShowBase.__init__(self)

        self.start = Node(x=start[0], y=start[1], z=start[2])
        self.goal = Node(x=goal[0], y=goal[1], z=goal[2])
        self.stepSize = stepSize
        self.maxIteration = iteration
        self.obstacles = obstacles
        self.nodeList = {}
        self.goalRadius = goalRadius

        # Initialize camera position and direction
        self.camera.setPos(0, -20, 0)
        self.camera.lookAt(0, 0, 0)

        # Set up lighting
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        self.loadObstacles()
        self.loadStartAndGoal()

    def loadObstacles(self):
        for obstacle in self.obstacles:
            x, y, z, r = obstacle
            sphere = self.loader.loadModel("models/smiley")
            sphere.setScale(r, r, r)
            sphere.setPos(x, y, z)
            sphere.setColor(0.5, 0.5, 0.5, 1)  # set color to red
            sphere.reparentTo(self.render)

    def loadStartAndGoal(self):
        start = self.loader.loadModel("models/smiley")
        start.setScale(10, 10, 10)
        start.setPos(*[self.start.x, self.start.y, self.start.z])
        start.setColor(0, 1, 0, 1)  # set color to green
        start.reparentTo(self.render)

        goal = self.loader.loadModel("models/smiley")
        goal.setScale(10, 10, 10)
        goal.setPos(*[self.goal.x, self.goal.y, self.goal.z])
        goal.setColor(0, 0, 1, 1)  # set color to blue
        goal.reparentTo(self.render)

    def loadPath(self, fromNode, toNode):
        x1, y1, z1 = fromNode.x, fromNode.y, fromNode.z
        x2, y2, z2 = toNode.x, toNode.y, toNode.z
        line = self.loader.loadModel("models/misc/smiley")
        line.setScale(2, 2, 2)
        line.setColor(1, 1, 0, 1)  # set color to yellow
        line.setPos(x1, y1, z1)
        lookAt = LVector3(x2 - x1, y2 - y1, z2 - z1)
        line.lookAt(lookAt)
        line.setHpr(line.getHpr() + LVector3(0, -90, 0))  # adjust orientation
        line.reparentTo(self.render)

    def drawSegment(self, p1, p2):
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]
        line = self.loader.loadModel("models/misc/smiley")
        line.setScale(2, 2, 2)
        line.setColor(0, 0, 1, 1)  # set color to blue
        line.setPos(x1, y1, z1)
        lookAt = LVector3(x2 - x1, y2 - y1, z2 - z1)
        line.lookAt(lookAt)
        line.setHpr(line.getHpr() + LVector3(0, -90, 0))  # adjust orientation
        line.reparentTo(self.render)

    def get_random_point(self):
        if random.randint(0, 100) > self.goalRadius:
            return [random.uniform(0, X_LIMIT), random.uniform(0, Y_LIMIT), random.uniform(0, Z_LIMIT)]
        return [self.goal.x, self.goal.y, self.goal.z]

    def dist_to_goal(self, x, y, z):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y, z - self.goal.z])

    def find_near_nodes(self, newNode, r=8):
        dList = [(key, (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 + (node.z - newNode.z) ** 2) for key, node
                 in self.nodeList.items()]
        nearNodesIndexes = [key for key, distance in dList if distance <= r ** 2]
        return nearNodesIndexes

    def get_nearest_node_index(self, nodeList, rnd):
        dList = [(key, (node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2) for key, node in
                 nodeList.items()]
        minind = min(dList, key=lambda d: d[1])
        return minind[0]

    def is_collision(self, node, obstacles):
        if obstacles:
            for o in obstacles:
                dx = o[0] - node.x
                dy = o[1] - node.y
                dz = o[2] - node.z
                d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if d <= o[3]:
                    return True
            return False
        return False

    def is_collision_after_connect(self, nearNode, theta, pitch, distance):
        tempNode = copy.deepcopy(nearNode)
        for i in range(int(distance / self.stepSize)):
            tempNode.x += self.stepSize * math.cos(theta) * math.cos(pitch)
            tempNode.y += self.stepSize * math.sin(theta) * math.cos(pitch)
            tempNode.z += self.stepSize * math.sin(pitch)
            if self.is_collision(tempNode, self.obstacles):
                return True
        return False

    def choose_parent(self, newNode, nearNodesIndexes):
        if len(nearNodesIndexes) == 0:
            return newNode
        distanceList = []
        for i in nearNodesIndexes:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            dz = newNode.z - self.nodeList[i].z
            dxy = math.sqrt(dx ** 2 + dy ** 2)
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            pitch = math.atan2(dz, dxy)
            theta = math.atan2(dy, dx)

            if not self.is_collision_after_connect(nearNode=self.nodeList[i], theta=theta, pitch=pitch, distance=d):
                distanceList.append(self.nodeList[i].cost + d)
            else:
                distanceList.append(float("inf"))

        min_cost = min(distanceList)
        if min_cost == float("inf"):
            return newNode
        newNode.cost = min_cost
        newNode.parentIndex = nearNodesIndexes[distanceList.index(min_cost)]
        return newNode

    def expand(self, randomNode, nearNodeIndex):
        """ Expand the tree at certain step from nearNode to randomNode
        :param randomNode: random node
        :param nearNodeIndex: index of the parent node
        :return:
        """
        nearNode = self.nodeList[nearNodeIndex]
        dx = randomNode[0] - nearNode.x
        dy = randomNode[1] - nearNode.y
        dz = randomNode[2] - nearNode.z
        dxy = math.sqrt(dx ** 2 + dy ** 2)
        pitch = math.atan2(dz, dxy)
        theta = math.atan2(dy, dx)
        newNode = copy.deepcopy(nearNode)
        newNode.x += self.stepSize * math.cos(theta) * math.cos(pitch)
        newNode.y += self.stepSize * math.sin(theta) * math.cos(pitch)
        newNode.z += self.stepSize * math.sin(pitch)
        newNode.cost += self.stepSize
        newNode.parentIndex = nearNodeIndex
        newNode.isLeaf = True
        return newNode

    def final_path(self, goalIndex):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        while self.nodeList[goalIndex].parentIndex is not None:
            node = self.nodeList[goalIndex]
            path.append([node.x, node.y, node.z])
            goalIndex = node.parentIndex
        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def get_best_last_index(self):
        distances_to_goal = [(key, self.dist_to_goal(node.x, node.y, node.z)) for key, node in self.nodeList.items()]
        nearGoalNodesIndexes = [key for key, distance in distances_to_goal if distance <= self.stepSize]

        if len(nearGoalNodesIndexes) == 0:
            return None

        min_cost = min([self.nodeList[key].cost for key in nearGoalNodesIndexes])
        for i in nearGoalNodesIndexes:
            if self.nodeList[i].cost == min_cost:
                print(min_cost)
                return i
        return None

    def update_near_nodes(self, newNodeIndex, newNode, nearNodesIndexes):
        """ Update the neighbourhood path
        :param newNodeIndex:
        :param newNode:
        :param nearNodesIndexes:
        :return:
        """
        for i in nearNodesIndexes:
            nearNode = self.nodeList[i]
            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            dz = newNode.z - nearNode.z
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            curr_cost = newNode.cost + d

            # check if new route is shorter
            if curr_cost < nearNode.cost:
                dxy = math.sqrt(dx ** 2 + dy ** 2)
                pitch = math.atan2(dz, dxy)
                theta = math.atan2(dy, dx)
                if not self.is_collision_after_connect(nearNode, theta, pitch, d):
                    self.nodeList[nearNode.parentIndex].isLeaf = True
                    for node in self.nodeList.values():
                        if node.parentIndex == nearNode.parentIndex and node != nearNode:
                            self.nodeList[nearNode.parentIndex].isLeaf = False
                            break
                    # update
                    nearNode.parentIndex = newNodeIndex
                    nearNode.cost = curr_cost
                    newNode.isLeaf = False
                    # print('rewired: ' + str(nearNode.x) + ', ' + str(nearNode.y) + ', ' + str(nearNode.z))

    def draw_graph(self, rnd=None):
        # self.loadObstacles()
        # self.loadStartAndGoal()
        for node in self.nodeList.values():
            if node.parentIndex:
                self.loadPath(self.nodeList[node.parentIndex], node)
        lastIndex = self.get_best_last_index()
        if lastIndex is not None:
            path = self.final_path(lastIndex)
            ind = len(path)
            while ind > 1:
                self.drawSegment(path[ind - 2], path[ind - 1])
                ind -= 1

    def run_RRT_Star(self, animation=True):
        self.nodeList[0] = self.start
        i = 0
        for i in range(self.maxIteration):
        # while True:
            i += 1
            rnd = self.get_random_point()
            nearNodeIndex = self.get_nearest_node_index(self.nodeList, rnd)
            newNode = self.expand(rnd, nearNodeIndex)
            if not self.is_collision(newNode, self.obstacles):
                nearNodesIndexes = self.find_near_nodes(newNode, 30)
                newNode = self.choose_parent(newNode, nearNodesIndexes)
                self.nodeList[newNode.parentIndex].isLeaf = False
                self.nodeList[i + 100] = newNode
                self.update_near_nodes(i + 100, newNode, nearNodesIndexes)
                if animation and i % 10 == 0:
                    self.draw_graph(rnd)
                # print(i)

            lastIndex = self.get_best_last_index()
            if lastIndex is None:
                continue
            path = self.final_path(lastIndex)
            return path


def main():
    print("start RRT* path planning")

    obstacleList = [
        (200, 150, 500, 60),
        (140, 220, 200, 60),
        (400, 12, 400, 100),
        (40, 20, 170, 170),
        (577, 700, 132, 60),
        (450, -175, -70, 100)
    ]

    start = [250, 350, 0]
    goal = [250, 0, 300]
    rrt = RRTStar(start=start, goal=goal, obstacles=obstacleList, stepSize=15, iteration=1000)
    path = rrt.run_RRT_Star(animation=show_animation)
    # rrt.run_RRT_Star(animation=show_animation)
    # print(path)
    rrt.run()


if __name__ == '__main__':
    main()