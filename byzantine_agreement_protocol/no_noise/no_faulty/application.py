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
    """
    Prepares the four-qubit singlet state using the Loop Circuit from Figure 6:
        (1 / 2√3)(2|0011⟩ - |0101⟩ - |0110⟩ - |1010⟩ - |1001⟩ + 2|1100⟩)
    """

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


class Sender(Program):
    PEER_R0 = "R0"
    PEER_R1 = "R1"

    def __init__(self, M: int = 5):
        self.M = M

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

        xS = random.choice([0, 1])
        checkSet = []

        for alpha in range(self.M):
            qubits = [Qubit(connection) for _ in range(4)]
            create_special_state(qubits)

            # Keep q0 and q1, send q2 and q3
            m0 = qubits[0].measure()
            m1 = qubits[1].measure()

            yield from teleport_send(qubits[2], context, peer_name=self.PEER_R0)
            yield from teleport_send(qubits[3], context, peer_name=self.PEER_R1)

            yield from context.connection.flush()

            if int(m0) == xS and int(m1) == xS:
                checkSet.append(alpha)

        csocket_r0.send((xS, checkSet))
        csocket_r1.send((xS, checkSet))

        return {"xS": xS, "m0": int(m0), "m1": int(m1), "checkSet": checkSet}


class Receiver0(Program):
    PEER_S = "S"
    PEER_R1 = "R1"

    def __init__(self, name: str, M: int = 5, mu: float = 0.272):
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

        # Invocation Phase
        for _ in range(self.M):
            received_qubit = yield from teleport_recv(context, peer_name=self.PEER_S)
            m_qubit = received_qubit.measure()

            yield from context.connection.flush()
            measurements.append(int(m_qubit))

        xS, checkSet = yield from csocket.recv()

        # Check Phase
        T = math.ceil(self.mu * self.M)
        if len(checkSet) >= T and all(measurements[alpha] != xS for alpha in checkSet):
            y = xS
        else:
            y = None

        # Cross-Calling Phase
        csocket_r1 = context.csockets[self.PEER_R1]
        csocket_r1.send((y, checkSet))

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

    def __init__(self, name: str, M: int = 5, mu: float = 0.272, lambda_: float = 0.94):
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

        # Invocation Phase
        for _ in range(self.M):
            received_qubit = yield from teleport_recv(context, peer_name=self.PEER_S)
            m_qubit = received_qubit.measure()

            yield from context.connection.flush()
            measurements.append(int(m_qubit))

        xS, checkSet = yield from csocket.recv()

        # Check Phase
        T = math.ceil(self.mu * self.M)
        if len(checkSet) >= T and all(measurements[alpha] != xS for alpha in checkSet):
            y = xS
        else:
            y = None

        # Cross-Calling Phase
        csocket_r0 = context.csockets[self.PEER_R0]
        y0, checkSet0 = yield from csocket_r0.recv()

        # Cross-Check Phase
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
