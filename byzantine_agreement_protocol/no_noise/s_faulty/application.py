import netsquid as ns
from netqasm.sdk import Qubit
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import teleport_send, teleport_recv
import random
import math

_ANGLE_RZ_Q0 = -0.73304
_ANGLE_RZ_Q2 = 2.67908
_ANGLE_RY_Q0 = -2.67908
_ANGLE_RZ_FINAL_Q2 = 1.5708

def create_special_state(qubits):
    if len(qubits) != 4:
        raise ValueError("Input must be a list of 4 Qubit objects.")

    q0, q1, q2, q3 = qubits

    q0.H()
    q1.H()
    q2.H()

    q0.rot_Z(angle=_ANGLE_RZ_Q0)
    q2.rot_Z(angle=_ANGLE_RZ_Q2)

    q2.cnot(q0)
    q2.H()

    q0.rot_Y(angle=_ANGLE_RY_Q0)

    q1.cnot(q0)
    q2.cnot(q3)

    q2.rot_Z(angle=_ANGLE_RZ_FINAL_Q2)

    q1.cnot(q3)
    q0.cnot(q2)


class FaultySender(Program):
    PEER_R0 = "R0"
    PEER_R1 = "R1"

    def __init__(self, M: int = 12, mu: float = 0.272, lambda_: float = 0.94):
        self.M = M
        self.mu = mu
        self.lambda_ = lambda_

    @property
    def meta(self):
        return ProgramMeta(
            name="sender_program",
            csockets=[self.PEER_R0, self.PEER_R1],
            epr_sockets=[self.PEER_R0, self.PEER_R1],
            max_qubits=4,
        )

    def run(self, context: ProgramContext):
        connection = context.connection
        csocket_r0 = context.csockets[self.PEER_R0]
        csocket_r1 = context.csockets[self.PEER_R1]

        T = math.ceil(self.mu * self.M)
        Q = T - math.ceil(self.lambda_ * T) + 1

        m0011 = []
        mmixed = []
        m1100 = []

        events = []
        for i in range(self.M):
            qubits = [Qubit(connection) for _ in range(4)]
            create_special_state(qubits)

            outcome0 = qubits[0].measure()
            outcome1 = qubits[1].measure()

            yield from context.connection.flush()

            outcome_pair = (int(outcome0), int(outcome1))

            yield from teleport_send(qubits[2], context, peer_name=self.PEER_R0)
            yield from teleport_send(qubits[3], context, peer_name=self.PEER_R1)
            yield from context.connection.flush()

            if outcome_pair == (0, 0):
                m0011.append(i)
            elif outcome_pair == (1, 1):
                m1100.append(i)
            else:
                mmixed.append(i)
            events.append((i, outcome_pair))

        l1 = len(m0011)
        l2 = len(mmixed)
        l3 = len(m1100)

        sigma0 = []
        sigma1 = []

        inDomain = True
        if T - Q <= l1 and Q <= l2 and T <= l3:
            sigma0 = m0011[:T - Q] + mmixed[:Q]
            sigma1 = m1100
        else:
            inDomain = False

        x0 = 0
        x1 = 1

        csocket_r0.send((x0, sigma0))
        csocket_r1.send((x1, sigma1))

        print(f"{ns.sim_time()} ns: Faulty Sender sent x0={x0}, x1={x1}, σ0={sigma0}, σ1={sigma1}")
        return {
            "inDomain": inDomain,
            "x0": x0,
            "x1": x1,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "classified": {
                "m0011": m0011,
                "mmixed": mmixed,
                "m1100": m1100,
            },
            "events": events
        }


class Receiver0(Program):
    PEER_S = "S"
    PEER_R1 = "R1"

    def __init__(self, name: str, M: int = 12, mu: float = 0.272):
        self.name = name
        self.M = M
        self.mu = mu

    @property
    def meta(self):
        return ProgramMeta(
            name="receiver_program",
            csockets=[self.PEER_S, self.PEER_R1],
            epr_sockets=[self.PEER_S],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        connection = context.connection
        csocket = context.csockets[self.PEER_S]

        measurements = []

        for _ in range(self.M):
            received_qubit = yield from teleport_recv(context, peer_name=self.PEER_S)
            m_qubit = received_qubit.measure()

            yield from context.connection.flush()
            measurements.append(int(m_qubit))

        xS, checkSet = yield from csocket.recv()

        T = math.ceil(self.mu * self.M)
        if len(checkSet) >= T and all(measurements[alpha] != xS for alpha in checkSet):
            y = xS
        else:
            y = None

        csocket_r1 = context.csockets[self.PEER_R1]
        csocket_r1.send((y, checkSet))

        print(f"{ns.sim_time()} ns: Receiver {self.name} measured: {measurements}, xS={xS}, checkSet={checkSet}, y={y}")
        return {
            "receiver": self.name,
            "measurements": measurements,
            "xS_received": xS,
            "checkSet": checkSet,
            "accepted": y,
        }


class Receiver1(Program):
    PEER_S = "S"
    PEER_R0 = "R0"

    def __init__(self, name: str, M: int = 12, mu: float = 0.272, lambda_: float = 0.94):
        self.name = name
        self.M = M
        self.mu = mu
        self.lambda_ = lambda_

    @property
    def meta(self):
        return ProgramMeta(
            name="receiver_program",
            csockets=[self.PEER_S, self.PEER_R0],
            epr_sockets=[self.PEER_S],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        connection = context.connection
        csocket = context.csockets[self.PEER_S]

        measurements = []

        for _ in range(self.M):
            received_qubit = yield from teleport_recv(context, peer_name=self.PEER_S)
            m_qubit = received_qubit.measure()

            yield from context.connection.flush()
            measurements.append(int(m_qubit))

        xS, checkSet = yield from csocket.recv()

        T = math.ceil(self.mu * self.M)
        if len(checkSet) >= T and all(measurements[alpha] != xS for alpha in checkSet):
            y = xS
        else:
            y = None

        csocket_r0 = context.csockets[self.PEER_R0]
        y0, checkSet0 = yield from csocket_r0.recv()

        if y is not None and y0 is not None and y != y0:
            required_matches = math.ceil(self.lambda_ * T + len(checkSet0) - T)
            count_matching = sum(1 for alpha in checkSet0 if measurements[alpha] == 1 - y0)

            if len(checkSet0) >= T and count_matching >= required_matches:
                y_final = y0
            else:
                y_final = y
        else:
            y_final = y

        return {
            "receiver": self.name,
            "measurements": measurements,
            "xS_received": xS,
            "checkSet": checkSet,
            "accepted": y,
            "final_decision": y_final,
        }
