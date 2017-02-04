import numpy as np
from numpy.linalg import inv, multi_dot
from numpy import dot
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math
import seaborn
from scipy import sparse
from scipy.sparse import linalg


def dots(*arg):
    return multi_dot(arg)


class BasicMovement:
    def __init__(self, maxSpeed, maxRotation, covariance, measureFunction):
        self.maxSpeed = maxSpeed
        self.maxRotation = maxRotation
        self.measureFunction = measureFunction
        self.covariance = np.atleast_2d(covariance)

    #  Input the real state
    def move(self, state, covariance=None, command=None):
        command = self.__choose_command(state) if command is None else command
        noise = self.__get_noise(covariance)
        idealMove = self.exact_move(state, command)
        realMove = self.__noisy_move(state, idealMove, noise)
        newState = state + realMove
        return clipState(newState), command

    def __choose_command(self, state):
        speed = self.maxSpeed * np.random.rand()
        if (np.linalg.norm(state[:2]) > 100):
            _, rotation = self.measureFunction(state[:3], [[0], [0]])
            rotation = np.clip(rotation, -self.maxRotation, self.maxRotation)
        else:
            rotation = (np.random.rand() * 2 - 1) * self.maxRotation
        return [speed, rotation]

    def exact_move(self, state, command):
        speed, rotation = command
        angle = state[2]
        deltaX = speed * math.cos(angle)
        deltaY = speed * math.sin(angle)

        move = np.zeros_like(state)
        move[:3, 0] = [deltaX, deltaY, rotation]
        return move

    def __noisy_move(self, state, idealMove, noise):
        noisyMove = idealMove[:3] + noise
        noisySpeed, _ = self.measureFunction(noisyMove[:3], np.zeros_like(noise)[:2])
        noisyRotation = noisyMove[2]

        maxs = [self.maxSpeed, self.maxRotation]
        mins = [0, -self.maxRotation]
        correctedCommand = np.clip([noisySpeed, noisyRotation], mins, maxs)
        return self.exact_move(state, correctedCommand)

    def __noisy_move2(self, state, idealMove, noise):
        noisyMove = np.zeros_like(state)
        noisyMove[:3] = idealMove[:3] + noise
        return noisyMove

    def __get_noise(self, covariance):
        covariance = self.covariance if covariance is None else covariance
        noise = np.random.multivariate_normal(np.zeros(covariance.shape[0]), covariance, 1).T
        return noise


class BasicMeasurement:
    def __init__(self, covariance, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize=0, detectionCone=0):
        self.covariance = np.atleast_2d(covariance)
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.measureFunction = measureFunction
        self.gradMeasureFunction = gradMeasureFunction
        self.detectionSize = detectionSize
        self.detectionCone = detectionCone

    #  Input the real state
    def measure(self, state):
        dim = state.shape[0]
        dimR = self.robotFeaturesDim
        dimE = self.envFeaturesDim
        rState = state[:dimR]
        envState = state[dimR:]
        nbLandmark = (dim - dimR) / dimE

        mes = np.zeros(nbLandmark * dimE).reshape(nbLandmark, dimE)
        landmarkIds = np.zeros(nbLandmark)
        j = 0

        for i, landmark in enumerate(envState.reshape((nbLandmark, dimE, 1))):
            diffNorm, diffAngle = self.measureFunction(rState, landmark)
            angleOk = (abs(clipAngle(diffAngle, True)) < self.detectionCone / 2.) or (self.detectionCone is 0)
            distanceOk = (diffNorm < self.detectionSize) or (self.detectionSize is 0)

            if distanceOk and angleOk:
                mes[j] = [diffNorm, diffAngle]
                landmarkIds[j] = i
                j += 1

        mes = mes[:j]
        landmarkIds = landmarkIds[:j]
        mes = np.array(mes) + self.__get_noise(mes)
        return mes, landmarkIds

    def __get_noise(self, mes):
        noise = np.random.multivariate_normal(np.zeros(self.covariance.shape[0]), self.covariance, mes.shape[0])
        return noise


class SEIFModel2:

    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial, maxLinks):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.mu = muInitial.copy()

        self.Sx = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.Sx[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel
        self.maxLinks = maxLinks

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update2(command, U)
        self.__mean_update()
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        self.__mean_update()
        self.__sparsification()
        return self.H, self.b, self.mu

    def __motion_update2(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        delta = dots(self.Sx.T, gradMeanMotion, self.Sx)
        G = dots(self.Sx, (inv(np.eye(r) + delta) - np.eye(r)), self.Sx.T)
        phi = np.eye(self.dimension) + G
        Hp = dots(phi.T, self.H, phi)
        deltaH = dots(Hp, self.Sx, inv(inv(U) + dots(self.Sx.T, Hp, self.Sx)), self.Sx.T, Hp)
        H = inv(Hp + dots(self.Sx, U, self.Sx.T))
        # H = Hp - deltaH
        # self.b = self.b - dot(previousMeanState.T, self.H - H) + dot(meanStateChange.T, H)
        self.H = H
        self.b = dot(newMeanState.T,  self.H)
        self.mu = newMeanState

    def __motion_update(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.mu
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)

        np.set_printoptions(precision=2, linewidth=200)
        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        delta = dots(self.Sx.T, gradMeanMotion, self.Sx)
        G = dots(self.Sx, (inv(np.eye(r) + delta) - np.eye(r)), self.Sx.T)
        phi1 = np.eye(self.dimension) + G
        phi = inv(np.eye(self.dimension) + gradMeanMotion)
        print "np.linalg.norm(phi1-phi)"
        print np.linalg.norm(phi1-phi)

        Hp = dots(phi.T, self.H, phi)
        deltaH = dots(Hp, self.Sx, inv(inv(U) + dots(self.Sx.T, Hp, self.Sx)), self.Sx.T, Hp)
        H = Hp - deltaH
        self.b = self.b - dot(previousMeanState.T, deltaH - self.H + Hp) + dot(meanStateChange.T, H)
        self.H = H


        print "_H_"
        print inv((Hp) + dots(self.Sx, U, self.Sx.T))
        print np.linalg.norm(inv(inv(Hp) + dots(self.Sx, U, self.Sx.T)) - eif.HH)
        print np.linalg.norm(inv(inv(Hp) + dots(self.Sx, U, self.Sx.T)) - self.H)
        print "self.H"
        print self.H
        print np.linalg.norm(self.H - eif.HH)
        print "eif.HH"
        print eif.HH


        # print "self.b"
        # print self.b
        # print eif.bb
        # print self.b - eif.bb
        # print np.linalg.norm(self.b - eif.bb)

    def __mean_update(self):
        ''' Coordinate ascent '''
        mu = self.mu
        iterations = 30
        y0, yp = self.__partition_links()
        y = np.concatenate([np.arange(self.robotFeaturesDim), y0, yp])

        # print self.H[:7]
        # print "self.b"
        # print self.b
        # print y0
        # print yp
        # print y
        # print "vrai mu"
        vMu = dot(self.b, inv(self.H)).T
        # print vMu
        muSave = []
        muSave2 = []

        # print mu[:3]
        for t in xrange(iterations):
            # print("\nmu %d" % t)
            for i in y:
                y2 = np.setdiff1d(y, i)
                mu[i] = (self.b[0, i] - dot(self.H[i, y2], mu[y2])) / self.H[i, i]
            # print mu[:3]
            # print np.linalg.norm(mu[:3] - vMu[:3])
            muSave.extend([np.linalg.norm(mu - vMu)])
            muSave2.extend([np.linalg.norm(mu - eif.estimate())])
        self.mu = mu
        # self.mu = vMu
        plt.plot(muSave)
        # plt.plot(muSave   2)
        # plt.show()

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.mu
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        correction = mesError + dot(C.T, mu)
        correction[1, 0] = clipAngle(correction[1, 0])
        self.H += dot(dot(C, self.invZ),  C.T)
        self.b += dot(dot(correction.T, self.invZ), C.T)

    def __partition_links(self):
        r = self.robotFeaturesDim
        e = self.envFeaturesDim
        d = self.dimension
        l = (d - r) / e
        arrRF = np.arange(r)

        norms = np.array([np.linalg.norm(self.H[arrRF][:, np.arange(i * e + r, (i + 1) * e + r)]) for i in xrange(l)])
        ids = np.argsort(norms)
        yp = ids[-self.maxLinks:]
        y0 = np.setdiff1d(np.where(norms > 0), yp)

        yp = np.concatenate([np.arange(y * e, (y + 1) * e) for y in yp]) + r
        if len(y0) > 0:
            y0 = np.concatenate([np.arange(y * e, (y + 1) * e) for y in y0]) + r

        return y0, yp

    def __build_projection_matrix(self, indices):
        d1 = self.H.shape[0]
        d2 = len(indices)

        S = np.zeros((d1, d2))
        S[indices] = np.eye(d2)
        return S

    def __sparsification(self):
        x = np.arange(self.robotFeaturesDim)
        y0, yp = self.__partition_links()
        Sx = self.__build_projection_matrix(x)
        Sy0 = self.__build_projection_matrix(y0)
        Sxy0 = self.__build_projection_matrix(np.concatenate((x, y0)))
        Sxyp = self.__build_projection_matrix(np.concatenate((x, yp)))
        Sxy0yp = self.__build_projection_matrix(np.concatenate((x, y0, yp)))

        # print("yp")
        # print(yp)
        # print("y0")
        # print(y0)
        # print(Sxy0yp)
        # print("self.H")
        # print(self.H)

        Hp = dots(Sxy0yp, Sxy0yp.T, self.H, Sxy0yp, Sxy0yp.T)
        # print("Hp")
        # print(Hp)
        # print(self.H - Hp)

        Ht = self.H - dots(Hp, Sy0, inv(dots(Sy0.T, Hp, Sy0)), Sy0.T, Hp) \
                    + dots(Hp, Sxy0, inv(dots(Sxy0.T, Hp, Sxy0)), Sxy0.T, Hp) \
                    - dots(self.H, Sx, inv(dots(Sx.T, self.H, Sx)), Sx.T, self.H)
        eps = 1e-5
        Ht[np.abs(Ht) < eps] = 0
        bt = self.b + dot(self.mu.T, Ht - self.H)

        self.H = Ht
        self.b = bt
        # print("bt")
        # print(bt)
        # print("Ht")
        # print(Ht)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self):
        return self.mu


class SEIFModel:

    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial, maxLinks):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.mu = muInitial.copy()

        self.Sx = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.Sx[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel
        self.maxLinks = maxLinks

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update_sparse(command, U)
        self.__mean_update()
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        self.__mean_update()
        self.__sparsification()
        return self.H, self.b, self.mu

    def __motion_update(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        delta = dots(self.Sx.T, gradMeanMotion, self.Sx)
        G = dots(self.Sx, (inv(np.eye(r) + delta) - np.eye(r)), self.Sx.T)
        phi = np.eye(self.dimension) + G
        Hp = dots(phi.T, self.H, phi)
        deltaH = dots(Hp, self.Sx, inv(inv(U) + dots(self.Sx.T, Hp, self.Sx)), self.Sx.T, Hp)
        # H = inv(Hp + dots(self.Sx, U, self.Sx.T))
        H = Hp - deltaH
        # self.b = self.b - dot(previousMeanState.T, self.H - H) + dot(meanStateChange.T, H)
        self.H = H
        self.b = dot(newMeanState.T,  self.H)
        self.mu = newMeanState


    def __motion_update_sparse(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        Sx = sparse.bsr_matrix(self.Sx)
        sH = sparse.bsr_matrix(self.H)
        invU = sparse.coo_matrix(inv(U))
        sparseGradMeanMotion = sparse.bsr_matrix(gradMeanMotion)

        delta = Sx.T.dot(sparseGradMeanMotion).dot(Sx)
        G = Sx.dot(linalg.inv(sparse.eye(r) + delta) - sparse.eye(r)).dot(Sx.T)
        phi = sparse.eye(self.dimension) + G
        Hp = phi.T.dot(sH).dot(phi)
        deltaH = Hp.dot(Sx).dot(linalg.inv(invU + Sx.T.dot(Hp).dot(Sx))).dot(Sx.T).dot(Hp)
        # H = inv(Hp + dots(self.Sx, U, self.Sx.T))
        H = Hp - deltaH
        # self.b = self.b - dot(previousMeanState.T, self.H - H) + dot(meanStateChange.T, H)
        self.H = H.todense()
        self.b = H.dot(newMeanState).T
        self.mu = newMeanState

    def __mean_update(self):
        ''' Coordinate ascent '''
        mu = self.mu
        iterations = 30
        y0, yp = self.__partition_links()
        y = np.concatenate([np.arange(self.robotFeaturesDim), y0, yp])

        # vMu = dot(self.b, inv(self.H)).T
        # muSave = []
        # muSave2 = []

        for t in xrange(iterations):
            for i in y:
                y2 = np.setdiff1d(y, i)
                mu[i] = (self.b[0, i] - dot(self.H[i, y2], mu[y2])) / self.H[i, i]
            # muSave.extend([np.linalg.norm(mu - vMu)])
        self.mu = mu
        # plt.plot(muSave)

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.mu
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        correction = mesError + dot(C.T, mu)
        correction[1, 0] = clipAngle(correction[1, 0])
        self.H += dot(dot(C, self.invZ),  C.T)
        self.b += dot(dot(correction.T, self.invZ), C.T)

    def __partition_links(self):
        r = self.robotFeaturesDim
        e = self.envFeaturesDim
        d = self.dimension
        l = (d - r) / e
        arrRF = np.arange(r)

        norms = np.array([np.linalg.norm(self.H[arrRF][:, np.arange(i * e + r, (i + 1) * e + r)]) for i in xrange(l)])
        ids = np.argsort(norms)
        yp = ids[-self.maxLinks:]
        y0 = np.setdiff1d(np.where(norms > 0), yp)

        yp = np.concatenate([np.arange(y * e, (y + 1) * e) for y in yp]) + r
        if len(y0) > 0:
            y0 = np.concatenate([np.arange(y * e, (y + 1) * e) for y in y0]) + r

        return y0, yp

    def __build_projection_matrix(self, indices):
        d1 = self.H.shape[0]
        d2 = len(indices)

        S = np.zeros((d1, d2))
        S[indices] = np.eye(d2)
        return S

    def __sparsification(self):
        x = np.arange(self.robotFeaturesDim)
        y0, yp = self.__partition_links()
        Sx = sparse.coo_matrix(self.__build_projection_matrix(x))
        Sy0 = sparse.coo_matrix(self.__build_projection_matrix(y0))
        Sxy0 = sparse.coo_matrix(self.__build_projection_matrix(np.concatenate((x, y0))))
        Sxyp = sparse.coo_matrix(self.__build_projection_matrix(np.concatenate((x, yp))))
        Sxy0yp = sparse.coo_matrix(self.__build_projection_matrix(np.concatenate((x, y0, yp))))
        H = sparse.bsr_matrix(self.H)

        Hp = Sxy0yp.dot(Sxy0yp.T).dot(H).dot(Sxy0yp).dot(Sxy0yp.T)

        Ht = H - (0 if not y0.size else Hp.dot(Sy0).dot(linalg.inv(Sy0.T.dot(Hp).dot(Sy0))).dot(Sy0.T).dot(Hp)) \
                + Hp.dot(Sxy0).dot(linalg.inv(Sxy0.T.dot(Hp).dot(Sxy0))).dot(Sxy0.T).dot(Hp) \
                - H.dot(Sx).dot(linalg.inv(Sx.T.dot(H).dot(Sx))).dot(Sx.T).dot(H)
        eps = 1e-5
        Htt = Ht.todense()
        Htt[np.abs(Htt) < eps] = 0
        bt = self.b + (Ht - H).dot(self.mu)

        self.H = Htt
        self.b = bt

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self):
        return self.mu


class EIFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.HH = np.eye(dimension)
        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.bb = dot(muInitial.T, self.H)
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        return self.H, self.b

    def __motion_update(self, command, U):
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        IA = np.eye(self.H.shape[0]) + gradMeanMotion  # TO IMPROVE
        sigma = dot(dot(IA, inv(self.H)), IA.T) + dot(dot(self.S, U), self.S.T)
        self.H = inv(sigma)
        self.b = dot((newMeanState).T,  self.H)
        self.HH = self.H.copy()
        self.bb = self.b.copy()

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.estimate()
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        mesError += dot(C.T, mu)
        mesError[1, 0] = clipAngle(mesError[1, 0])
        self.H += dot(dot(C, self.invZ),  C.T)
        self.b += dot(dot(mesError.T, self.invZ), C.T)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self, H=None, b=None):
        H = self.H if H is None else H
        b = self.b if b is None else b
        return clipState(dot(b, inv(H)).T)


class EKFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.Sigma = np.eye(dimension)
        self.mu = muInitial.copy()
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.Z = covMes
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        return self.Sigma, self.mu

    def __motion_update(self, command, U):
        previousMeanState = self.mu
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.Sigma)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        IA = np.eye(self.Sigma.shape[0]) + gradMeanMotion  # TO IMPROVE
        self.mu = newMeanState
        self.Sigma = dot(dot(IA, self.Sigma), IA.T) + dot(dot(self.S, U), self.S.T)

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.mu
        Sigma = self.Sigma
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        toInvert = inv(dot(dot(C.T, Sigma), C) + self.Z)
        K = dot(dot(Sigma, C), toInvert)

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        mesError = dot(K, mesError)
        mesError[1, 0] = clipAngle(mesError[1, 0])

        self.mu += mesError
        self.Sigma = dot(np.eye(self.dimension) - dot(K, C.T), Sigma)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self):
        return self.mu


def measureFunction(rState, landmark):
    rDim = 3
    diff = landmark - rState[:rDim-1]
    diffNorm = np.linalg.norm(diff)

    angle = rState[rDim-1, 0]
    diffAngle = math.atan2(diff[1], diff[0]) - angle
    diffAngle = clipAngle(diffAngle)

    return diffNorm, diffAngle


def gradMeasureFunction(rState, landmark, ldmIndex):
    rDim = 3
    eDim = 2
    diff = (rState[:rDim-1] - landmark).flatten()
    diffNorm = np.linalg.norm(diff)

    grad = np.zeros(dimension * 2).reshape(dimension, 2)
    grad[:rDim-1, 0] = diff / diffNorm
    grad[ldmIndex:ldmIndex + eDim, 0] = -grad[:rDim-1, 0]
    grad[:rDim-1, 1] = np.array([-diff[1], diff[0]]) / (diffNorm**2)
    grad[ldmIndex:ldmIndex + eDim, 1] = -grad[:rDim-1, 1]
    grad[rDim-1, 1] = -1

    return grad


def clipAngle(angle, force=False):
    if clip or force:
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle


def clipState(state):
    if clip:
        state[2, 0] = clipAngle(state[2, 0])
    return state

clip = False


if __name__ == '__main__':

    dimension = None

    def simu():
        global dimension
        T = 100  # Number of timesteps
        nbLandmark = 900
        maxSpeed = 5
        maxRotation = 45 * math.pi / 180  # 45  # en radians
        sizeMap = 50

        # Robot Detection Parameters
        detectionSize = 2  # 40
        detectionCone = 180 * math.pi / 180  # en radians

        # Dimension Constants
        robotFeaturesDim = 3
        envFeaturesDim = 2
        commandsDim = 2
        mesDim = 2
        dimension = robotFeaturesDim + nbLandmark * envFeaturesDim

        # Covariances for motions and measurements
        covarianceMotion = np.eye(robotFeaturesDim)
        covarianceMotion[0, 0] = 1 ** 2  # motion noise variance X
        covarianceMotion[1, 1] = 1 ** 2  # motion noise variance Y
        covarianceMotion[2, 2] = (5 * math.pi / 180) ** 2  # motion noise variance Angle

        covarianceMeasurements = np.eye(mesDim)
        covarianceMeasurements[0, 0] = 1 ** 2  # measurement noise variance distance
        covarianceMeasurements[1, 1] = (5 * math.pi / 180) ** 2  # motion noise variance Angle


        ## ----------------------------------------------------------------------
        ## Simulation initialization

        ## -------------------
        ## State Definition

        # Real robot state
        state = np.zeros((dimension, 1))

        x = np.linspace(-sizeMap, sizeMap, np.sqrt(nbLandmark))
        y = np.linspace(-sizeMap, sizeMap, np.sqrt(nbLandmark))
        xv, yv = np.meshgrid(x, y)
        state[robotFeaturesDim:, 0] = np.vstack([xv.ravel(), yv.ravel()]).ravel(order="F")
        # state[robotFeaturesDim:] = np.random.rand(nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1) * 300 - 150


        # Basic and EIF estimator for robot state
        mu = state.copy()
        mu[robotFeaturesDim:] += np.random.normal(0, covarianceMeasurements[0, 0], nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1)
        muEKF = mu.copy()
        muEIF = mu.copy()
        muSEIF = mu.copy()

        ## --------------------
        ## Models Definition

        motionModel = BasicMovement(maxSpeed, maxRotation, covarianceMotion, measureFunction)
        measurementModel = BasicMeasurement(covarianceMeasurements, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize, detectionCone)
        ekf = EKFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)
        eif = EIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)
        seif = SEIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu, 4)
        # seif2 = SEIFModel2(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu, 2)

        mus_simple = np.zeros((T, dimension))
        mus_ekf = np.zeros((T, dimension))
        mus_eif = np.zeros((T, dimension))
        mus_seif = np.zeros((T, dimension))
        # mus_seif2 = np.zeros((T, dimension))
        states = np.zeros((T, dimension))

        mus_simple[0] = np.squeeze(mu)
        mus_ekf[0] = np.squeeze(muEKF)
        mus_eif[0] = np.squeeze(muEIF)
        mus_seif[0] = np.squeeze(muEIF)
        # mus_seif2[0] = np.squeeze(muEIF)
        states[0] = np.squeeze(state)


        # LOG Initial state
        # print("BEFORE")
        # print("EIF estimate :")
        # print(muEIF)
        # print("EKF estimate :")
        # print(muEKF)
        # print("Real state :")
        # print(state)
        # print('\n')

        for t in range(1, T):
            print("\nIteration %d" % t)
            state, motionCommand = motionModel.move(state)
            measures, landmarkIds = measurementModel.measure(state)

            mu += motionModel.exact_move(mu, motionCommand)

            # H, _ = ekf.update(measures, landmarkIds, motionCommand, covarianceMotion)
            # print (H != 0).sum(), ' / ', H.size
            # H, _, _ = eif.update(measures, landmarkIds, motionCommand, covarianceMotion)
            # print (H != 0).sum(), ' / ', H.size
            H, _, _ = seif.update(measures, landmarkIds, motionCommand, covarianceMotion)
            print (H != 0).sum(), ' / ', H.size
            # H, _, _ = seif2.update(measures, landmarkIds, motionCommand, covarianceMotion)
            # print (H != 0).sum(), ' / ', H.size

            # muEKF = ekf.estimate()
            # muEIF = eif.estimate()
            muSEIF = seif.estimate()
            # muSEIF2 = seif2.estimate()

            # print "np.linalg.norm(muEIF-muSEIF)"
            # print np.linalg.norm(muEIF-muSEIF)
            # print np.linalg.norm(eif.b - seif.b)
            # print np.linalg.norm(eif.H - seif.H)
            # print muEIF[:3]
            # print muSEIF[:3]


            mus_simple[t] = np.squeeze(mu)
            # mus_ekf[t] = np.squeeze(muEKF)
            # mus_eif[t] = np.squeeze(muEIF)
            mus_seif[t] = np.squeeze(muSEIF)
            # mus_seif2[t] = np.squeeze(muSEIF2)
            states[t] = np.squeeze(state)


        # # LOG Final state
        # print('\n')
        # print('AFTER')
        # print("EIF estimate :")
        # print(muEIF)
        # # print("EKF estimate :")
        # # print(muEKF)
        # print("Real state :")
        # print(state)
        # print("Final Error EIF:")
        # print(state - muEIF)
        # # print("Final Error EKF:")
        # # print(state - muEKF)
        # print("Final Max Error EIF: %f" % max(state-muEIF))
        # print("Final Norm Error EIF: %f" % np.linalg.norm(state-muEIF))
        # # print("Final Max Error EKF: %f" % max(state-muEKF))
        # # print("Final Norm Error EKF: %f" % np.linalg.norm(state-muEKF))
        # print("Final Max Error SEIF: %f" % max(state-muSEIF))
        # print("Final Norm Error SEIF: %f" % np.linalg.norm(state-muSEIF))

        landmarks = state[robotFeaturesDim:].reshape(nbLandmark, 2)
        plt.figure()
        ax = plt.gca()
        for x, y in landmarks:
            ax.add_artist(Circle(xy=(x, y),
                          radius=detectionSize,
                          alpha=0.3))
        plt.scatter(landmarks[:, 0], landmarks[:, 1])

        plt.plot(states[:, 0], states[:, 1])
        plt.plot(mus_simple[:, 0], mus_simple[:, 1])
        # plt.plot(mus_ekf[:, 0], mus_ekf[:, 1])
        # plt.plot(mus_eif[:, 0], mus_eif[:, 1])
        plt.plot(mus_seif[:, 0], mus_seif[:, 1])
        # plt.plot(mus_seif2[:, 0], mus_seif2[:, 1])

        # plt.legend(['Real position', 'Simple estimate', 'EKF estimate', 'EIF estimate', 'SEIF estimate'])
        plt.legend(['Real position', 'Simple estimate', 'EKF estimate', 'SEIF estimate', 'SEIF2 estimate'])
        plt.title("{0} landmarks".format(nbLandmark))
        plt.show()

    import cProfile
    cProfile.run('simu()')
