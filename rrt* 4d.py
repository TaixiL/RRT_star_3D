import random
import math
import copy
import sys
import pygame
import timeit
import numpy as np

show_animation = True

X_LIMIT = 100
Y_LIMIT = 100
Z_LIMIT = 100
T_LIMIT = 100


# windowSize = [X_LIMIT, Y_LIMIT, Z_LIMIT]

# pygame.init()
# fpsClock = pygame.time.Clock()

# screen = pygame.display.set_mode(windowSize)
# screen.fill((255, 255, 255))
# pygame.display.set_caption("RRT*")


class Node:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.parentIndex = None
        self.isLeaf = True
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, obstacles, stepSize=5.0, iteration=500, goalRadius=10):
        """ RRT* initiation
        :param start: start position
        :param goal: goal position
        :param obstacles: obstacle list
        :param stepSize: process step
        :param iteration: loop limit
        """
        self.start = Node(x=start[0], y=start[1], z=start[2], t=start[3])
        self.goal = Node(x=goal[0], y=goal[1], z=goal[2], t=goal[3])
        self.stepSize = stepSize
        self.maxIteration = iteration
        self.obstacles = obstacles
        self.nodeList = {}
        self.goalRadius = goalRadius

    def get_random_point(self):
        if random.randint(0, 100) > self.goalRadius:
            return [random.uniform(-X_LIMIT, X_LIMIT), random.uniform(-Y_LIMIT, Y_LIMIT),
                    random.uniform(-Z_LIMIT, Z_LIMIT), random.uniform(-T_LIMIT, T_LIMIT)]
        return [self.goal.x, self.goal.y, self.goal.z, self.goal.t]

    def dist_to_goal(self, x, y, z, t):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y, z - self.goal.z, t - self.goal.t])

    def find_near_nodes(self, newNode, r=8):
        dList = [(key, (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 + (node.z - newNode.z) ** 2 + (
                    node.t - newNode.t) ** 2) for key, node in self.nodeList.items()]
        nearNodesIndexes = [key for key, distance in dList if distance <= r ** 2]
        return nearNodesIndexes

    def get_nearest_node_index(self, nodeList, rnd):
        dList = [
            (key, (node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2 + (node.t - rnd[3]) ** 2) for
            key, node in nodeList.items()]
        minind = min(dList, key=lambda d: d[1])
        return minind[0]

    def is_collision(self, node, obstacles):
        if obstacles:
            for o in obstacles:
                dx = o[0] - node.x
                dy = o[1] - node.y
                dz = o[2] - node.z
                dt = o[3] - node.t
                d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + dt ** 2)
                if d <= o[3]:
                    return True
            return False
        return False

    def is_collision_after_connect(self, nearNode, theta, pitch, distance):
        tempNode = copy.deepcopy(nearNode)
        for i in range(int(distance / self.stepSize)):
            tempNode.x += self.stepSize * math.cos(theta)*math.cos(pitch)
            tempNode.y += self.stepSize *math.sin(pitch)
            tempNode.z += self.stepSize * -math.sin(pitch)*math.cos(theta)
            tempNode.t+=self.stepSize*np.random.rand()*10

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
            dt=newNode.t-self.nodeList[i].t
            dxy = math.sqrt(dx ** 2 + dy ** 2)
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2+dt**2)
            pitch = math.atan2(math.atan2(dz, dxy),dt)
            theta = math.atan2(dy, dx)

            # check if the connection of near nodes with new node has collision
            #if not self.is_collision_after_connect(nearNode=self.nodeList[i], theta=theta, pitch=pitch, distance=d):
            if not self.colision_vevtorBase(newNode=self.nodeList[i],oldNode=newNode):
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
        dt=randomNode[3]-nearNode.t
        dxy = math.sqrt(dx ** 2 + dy ** 2)
        pitch = math.atan2(dz, dxy)
        theta = math.atan2(dy, dx)
        newNode = copy.deepcopy(nearNode)
        newNode.x += self.stepSize*math.cos(theta)
        newNode.y += self.stepSize*math.sin(theta)
        newNode.z += self.stepSize* (randomNode[2] - nearNode.z) / math.sqrt(
                (randomNode[0] - nearNode.x) ** 2 + (randomNode[1] - nearNode.y) ** 2+0.1)
        nearNode.t+=self.stepSize*(randomNode[3] - nearNode.t) / math.sqrt(
                (randomNode[0] - nearNode.x) ** 2 + (randomNode[1] - nearNode.y) ** 2+0.1)
        newNode.cost += self.stepSize
        newNode.parentIndex = nearNodeIndex
        newNode.isLeaf = True
        return newNode

    def final_path(self, goalIndex):
        path = [[self.goal.x, self.goal.y, self.goal.z,self.goal.t]]
        while self.nodeList[goalIndex].parentIndex is not None:
            node = self.nodeList[goalIndex]
            path.append([node.x, node.y, node.z,node.t])
            goalIndex = node.parentIndex
        path.append([self.start.x, self.start.y, self.start.z,self.start.t])
        return path

    def get_best_last_index(self):
        distances_to_goal = [(key, self.dist_to_goal(node.x, node.y, node.z,node.t)) for key, node in self.nodeList.items()]
        nearGoalNodesIndexes = [key for key, distance in distances_to_goal if distance <= self.stepSize]

        if len(nearGoalNodesIndexes) == 0:
            return None

        min_cost = min([self.nodeList[key].cost for key in nearGoalNodesIndexes])
        for i in nearGoalNodesIndexes:
            if self.nodeList[i].cost == min_cost:
                # print(min_cost)
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
            dt=newNode.t-nearNode.t
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2+dt**2)
            curr_cost = newNode.cost + d

            # check if new route is shorter
            if curr_cost < nearNode.cost:
                dxy = math.sqrt(dx ** 2 + dy ** 2)
                pitch = math.atan2(dz, dxy)
                theta = math.atan2(dy, dx)
                #if not self.is_collision_after_connect(nearNode, theta, pitch, d):
                if not self.colision_vevtorBase(nearNode,newNode):
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

    def colision_vevtorBase(self,newNode,oldNode):
        tempNode = copy.deepcopy(oldNode)
        dx=newNode.x-oldNode.x
        dy=newNode.y-oldNode.y
        dz=newNode.z-oldNode.z
        dt=newNode.t-oldNode.t
        for i in range(100):
            tempNode.x += self.stepSize/100*dx
            tempNode.y += self.stepSize/100 *dy
            tempNode.z += self.stepSize/100*dz
            tempNode.t += self.stepSize/100*dt

            if self.is_collision(tempNode, self.obstacles):
                return True
        return False
    # def draw_graph(self, rnd=None):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # for node in self.nodeList.values():
    #     #     if node.parentIndex:
    #     #         pygame.draw.line(screen, (0, 255, 0),
    #     #                          [self.nodeList[node.parentIndex].x, self.nodeList[node.parentIndex].y],
    #     #                          [node.x, node.y])
    #     for node in self.nodeList.values():
    #         if node.parentIndex:
    #             x_vals = [self.nodeList[node.parentIndex].x, node.x]
    #             y_vals = [self.nodeList[node.parentIndex].y, node.y]
    #             z_vals = [self.nodeList[node.parentIndex].z, node.z]
    #             ax.plot(x_vals, y_vals, z_vals, c=(0, 1, 0))
    #
    #     for node in self.nodeList.values():
    #         if node.isLeaf:
    #             ax.scatter(node.x, node.y, node.z, marker='o', s=0.1, c=(1, 0, 1))
    #             pass
    #
    #     # pygame.draw.circle(screen, (255, 0, 0), [self.start.x, self.start.y], 10)
    #     # pygame.draw.circle(screen, (0, 0, 255), [self.goal.x, self.goal.y], 10)
    #     ax.scatter(self.start.x, self.start.y, self.start.z, marker='o', s=100, c=(1, 0, 0))
    #     ax.scatter(self.goal.x, self.goal.y, self.goal.z, marker='o', s=100, c=(0, 0, 1))
    #
    #     for obstacle in self.obstacles:
    #         x, y, z, r = obstacle
    #         u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    #         x_vals = r * np.cos(u) * np.sin(v) + x
    #         y_vals = r * np.sin(u) * np.sin(v) + y
    #         z_vals = r * np.cos(v) + z
    #         ax.plot_surface(x_vals, y_vals, z_vals, color=(0.5, 0.5, 0.5))
    #
    #     lastIndex = self.get_best_last_index()
    #     if lastIndex is not None:
    #         path = self.final_path(lastIndex)
    #         # ind = len(path)
    #         # while ind > 1:
    #         # pygame.draw.line(screen, (150, 150, 150), path[ind - 2], path[ind - 1], width=2)
    #         # ind -= 1
    #         pathPoints = path
    #         for i in range(len(pathPoints) - 1):
    #             x_vals = [pathPoints[i][0], pathPoints[i + 1][0]]
    #             y_vals = [pathPoints[i][1], pathPoints[i + 1][1]]
    #             z_vals = [pathPoints[i][2], pathPoints[i + 1][2]]
    #             ax.plot(x_vals, y_vals, z_vals, c='b', linewidth=2)
    #
    #         # set labels for the axes
    #         ax.set_xlabel('X axis')
    #         ax.set_ylabel('Y axis')
    #         ax.set_zlabel('Z axis')
    #
    #         # show the plot
    #         plt.show()
    #         # plt.close()

    def run_RRT_Star(self, animation=True):
        self.nodeList[0] = self.start
        i = 0
        for i in range(self.maxIteration):
            # while True:
            # print(i)
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
                # if animation and i % 10 == 0:
                #     self.draw_graph(rnd)
                # for e in pygame.event.get():
                #     if e.type == pygame.QUIT or (e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE):
                #         sys.exit("Exiting")
                # print(i)
            lastIndex = self.get_best_last_index()
            if lastIndex is None:
                continue
            path = self.final_path(lastIndex)
            # print(path[len(path)-1][3],"wohi")
            # path[len(path)-1][3]=self.goal.t
        return path


def main():
    print("start RRT* path planning")

    obstacleList = [
        (200, 150, 500, 60,1),
        (100, 300, 500, 60,1),
        (70,70,500,60,1),
        (300, 300, 500, 60,1),
    ]

    # Set Initial parameters
    start = [250, 350, 0,0]
    goal = [100, 350, 10,300]
    rrt = RRTStar(start=start, goal=goal, obstacles=obstacleList, stepSize=15, iteration=3000)
    path = rrt.run_RRT_Star(animation=show_animation)
    # rrt.run_RRT_Star(animation=show_animation)
    # print(rrt.nodeList)
    path.append(start)
    path.reverse()
    for i in path:
        print(i)
    #print(path)
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
