import random
import math
import copy
import sys
import pygame
import timeit
import numpy as np
import matplotlib.pyplot as plt

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


class BRRTStar:
    def __init__(self, start, goal, obstacles, stepSize=5.0, iteration=500, goalRadius=10):
        """ RRT* initiation
        :param start: start position
        :param goal: goal position
        :param obstacles: obstacle list
        :param stepSize: process step
        :param iteration: loop limit
        """
        self.start = Node(x=start[0], y=start[1], z=start[2])
        self.goal = Node(x=goal[0], y=goal[1], z=goal[2])
        self.stepSize = stepSize
        self.maxIteration = iteration
        self.obstacles = obstacles
        self.nodeList = {}
        self.nodeList_goal = {}
        self.goalRadius = goalRadius

    def get_random_point(self):
        if random.randint(0, 100) > self.goalRadius:
            return [random.uniform(-X_LIMIT, X_LIMIT), random.uniform(-Y_LIMIT, Y_LIMIT), random.uniform(-Z_LIMIT, Z_LIMIT)]
        return [self.goal.x, self.goal.y, self.goal.z]

    def dist_to_goal(self, x, y, z, x_goal, y_goal, z_goal):
        return np.linalg.norm([x - x_goal, y - y_goal, z - z_goal])

    def find_near_nodes(self, newNode,nodeList, r=8):
        dList = [(key, (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 + (node.z - newNode.z) ** 2) for key, node in nodeList.items()]
        nearNodesIndexes = [key for key, distance in dList if distance <= r ** 2]
        return nearNodesIndexes

    def get_nearest_node_index(self, nodeList, rnd):
        dList = [(key, (node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2) for key, node in nodeList.items()]
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

    def choose_parent(self, newNode, nearNodesIndexes, nodeList):
        if len(nearNodesIndexes) == 0:
            return newNode
        distanceList = []
        for i in nearNodesIndexes:
            dx = newNode.x - nodeList[i].x
            dy = newNode.y - nodeList[i].y
            dz = newNode.z - nodeList[i].z
            dxy = math.sqrt(dx ** 2 + dy ** 2)
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            pitch = math.atan2(dz, dxy)
            theta = math.atan2(dy, dx)

            # check if the connection of near nodes with new node has collision
            if not self.is_collision_after_connect(nearNode=nodeList[i], theta=theta, pitch=pitch, distance=d):
                distanceList.append(nodeList[i].cost + d)
            else:
                distanceList.append(float("inf"))

        min_cost = min(distanceList)
        if min_cost == float("inf"):
            return newNode
        newNode.cost = min_cost
        newNode.parentIndex = nearNodesIndexes[distanceList.index(min_cost)]
        return newNode

    def expand(self, randomNode, nearNodeIndex, nodeList):
        """ Expand the tree at certain step from nearNode to randomNode
        :param randomNode: random node
        :param nearNodeIndex: index of the parent node
        :return:
        """
        nearNode = nodeList[nearNodeIndex]
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

    def final_path(self, startIndex, nodeList, nodeList_goal, goalIndex):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        while nodeList[startIndex].parentIndex is not None:
            node = nodeList[startIndex]
            path.append([node.x, node.y, node.z])
            startIndex = node.parentIndex
        while nodeList_goal[goalIndex].parentIndex is not None:
            node = nodeList[goalIndex]
            path.append([node.x, node.y, node.z])
            goalIndex = node.parentIndex
        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def get_best_last_index(self, nodeList, nodeList_goal):
        distances_to_goal = [(key, key_goal, self.dist_to_goal(node.x, node.y, node.z, node_goal.x, node_goal.y, node_goal.z))
                             for key_goal, node_goal in nodeList_goal.items()
                             for key, node in nodeList.items()]

        nearGoalNodesIndexes = [(key, key_goal) for key, key_goal, distance in distances_to_goal if distance <= self.stepSize]

        if len(nearGoalNodesIndexes) == 0:
            return None
        min_cost = min([nodeList[key].cost+nodeList_goal[key_goal].cost for key, key_goal in nearGoalNodesIndexes])
        for i in nearGoalNodesIndexes:
            if nodeList[i[0]].cost + nodeList[i[1]].cost == min_cost:
                return i
        return None

    def update_near_nodes(self, newNodeIndex, newNode, nearNodesIndexes, nodeList):
        """ Update the neighbourhood path
        :param newNodeIndex:
        :param newNode:
        :param nearNodesIndexes:
        :return:
        """
        for i in nearNodesIndexes:
            nearNode = nodeList[i]
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
                    nodeList[nearNode.parentIndex].isLeaf = True
                    for node in nodeList.values():
                        if node.parentIndex == nearNode.parentIndex and node != nearNode:
                            nodeList[nearNode.parentIndex].isLeaf = False
                            break
                    # update
                    nearNode.parentIndex = newNodeIndex
                    nearNode.cost = curr_cost
                    newNode.isLeaf = False
                    # print('rewired: ' + str(nearNode.x) + ', ' + str(nearNode.y) + ', ' + str(nearNode.z))

    def draw_graph(self,nodeList, nodeList_goal, rnd=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # for node in self.nodeList.values():
        #     if node.parentIndex:
        #         pygame.draw.line(screen, (0, 255, 0),
        #                          [self.nodeList[node.parentIndex].x, self.nodeList[node.parentIndex].y],
        #                          [node.x, node.y])
        for node in nodeList.values():
            if node.parentIndex:
                x_vals = [nodeList[node.parentIndex].x, node.x]
                y_vals = [nodeList[node.parentIndex].y, node.y]
                z_vals = [nodeList[node.parentIndex].z, node.z]
                ax.plot(x_vals, y_vals, z_vals, color=(0, 1, 0))

        for node in nodeList.values():
            if node.isLeaf:
                ax.scatter(node.x, node.y, node.z, marker='o', s=0.1, color=(1, 0, 1))
                pass

        for node in nodeList_goal.values():
            if node.parentIndex:
                x_vals = [nodeList_goal[node.parentIndex].x, node.x]
                y_vals = [nodeList_goal[node.parentIndex].y, node.y]
                z_vals = [nodeList_goal[node.parentIndex].z, node.z]
                ax.plot(x_vals, y_vals, z_vals, color=(0, 1, 0))

        for node in nodeList_goal.values():
            if node.isLeaf:
                ax.scatter(node.x, node.y, node.z, marker='o', s=0.1, color=(1, 0, 1))
                pass

        # pygame.draw.circle(screen, (255, 0, 0), [self.start.x, self.start.y], 10)
        # pygame.draw.circle(screen, (0, 0, 255), [self.goal.x, self.goal.y], 10)
        ax.scatter(self.start.x, self.start.y, self.start.z, marker='o', s=100, color=(1, 0, 0))
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, marker='o', s=100, color=(0, 0, 1))

        for obstacle in self.obstacles:
            x, y, z, r = obstacle
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x_vals = r * np.cos(u) * np.sin(v) + x
            y_vals = r * np.sin(u) * np.sin(v) + y
            z_vals = r * np.cos(v) + z
            ax.plot_surface(x_vals, y_vals, z_vals, color=(0.5, 0.5, 0.5))

        lastIndex = self.get_best_last_index(nodeList, nodeList_goal)
        if lastIndex is not None:
            path = self.final_path(lastIndex[0], nodeList, nodeList_goal, lastIndex[1])
            # ind = len(path)
            # while ind > 1:
                # pygame.draw.line(screen, (150, 150, 150), path[ind - 2], path[ind - 1], width=2)
                # ind -= 1
            pathPoints = path
            for i in range(len(pathPoints) - 1):
                x_vals = [pathPoints[i][0], pathPoints[i + 1][0]]
                y_vals = [pathPoints[i][1], pathPoints[i + 1][1]]
                z_vals = [pathPoints[i][2], pathPoints[i + 1][2]]
                ax.plot(x_vals, y_vals, z_vals, c='b', linewidth=2)

            # set labels for the axes
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            # show the plot
            plt.show()
            # plt.close()

    def run_RRT_Star(self, animation=True):
        self.nodeList[0] = self.start
        self.nodeList_goal[0] = self.goal
        i = 0
        for i in range(self.maxIteration):
        # while True:
            i += 1
            rnd = self.get_random_point()
            nearNodeIndex = self.get_nearest_node_index(self.nodeList, rnd)
            newNode = self.expand(rnd, nearNodeIndex, self.nodeList)
            if not self.is_collision(newNode, self.obstacles):
                nearNodesIndexes = self.find_near_nodes(newNode, self.nodeList, 30)
                newNode = self.choose_parent(newNode, nearNodesIndexes, self.nodeList)
                self.nodeList[newNode.parentIndex].isLeaf = False
                self.nodeList[i + 100] = newNode
                self.update_near_nodes(i + 100, newNode, nearNodesIndexes, self.nodeList)
                # if animation and i % 10 == 0:
                #     self.draw_graph(self.nodeList, rnd)

            rnd_goal = self.get_random_point()
            nearNodeIndex_goal = self.get_nearest_node_index(self.nodeList_goal, rnd_goal)
            newNode_goal= self.expand(rnd_goal, nearNodeIndex_goal, self.nodeList_goal)
            if not self.is_collision(newNode_goal, self.obstacles):
                nearNodesIndexes_goal = self.find_near_nodes(newNode_goal, self.nodeList_goal, 30)
                newNode_goal = self.choose_parent(newNode_goal, nearNodesIndexes_goal, self.nodeList_goal)
                self.nodeList_goal[newNode_goal.parentIndex].isLeaf = False
                self.nodeList_goal[i + 100] = newNode_goal
                self.update_near_nodes(i + 100, newNode_goal, nearNodesIndexes_goal, self.nodeList_goal)
            if animation and i % 10 == 0:
                self.draw_graph(self.nodeList, self.nodeList_goal, rnd_goal)

            lastIndex = self.get_best_last_index(self.nodeList, self.nodeList_goal)
            if lastIndex is None:
                continue
            path = self.final_path(lastIndex[0], self.nodeList, self.nodeList_goal, lastIndex[1])
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

    # Set Initial parameters
    start = [250, 350, 0]
    goal = [250, -100, 300]
    rrt = BRRTStar(start=start, goal=goal, obstacles=obstacleList, stepSize=15, iteration=5000)
    path = rrt.run_RRT_Star(animation=show_animation)
    # rrt.run_RRT_Star(animation=show_animation)
    # print(rrt.nodeList)
    print(path)
    # rrt.draw_graph()
    # if show_animation:
    #     # rrt.draw_graph()
    #     # ind = len(path)
    #     # while ind > 1:
    #     #     pygame.draw.line(screen, (255, 0, 0), path[ind - 2], path[ind - 1], width=3)
    #     #     ind -= 1
    #     while True:
    #         for e in pygame.event.get():
    #             if e.type == pygame.QUIT or (e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE):
    #                 sys.exit("Exiting")


if __name__ == '__main__':
    main()

