class Executor:
    # What exactly does this class do?
    # Is it only useful for actual real execution, not simulations?
    #
    # port, host... localhost? only needed when network is...?
    # perhaps pass a [virtual] connection object instead!
    # what is the protocol of the object? TCP/UDP? What module node should
    # each one go to? idk... assume single for now.
    #
    # self.conn = conn # Don't handle actual connecting/etc inside class!
    # just provide an interface for STREAMS or PACKETS?
    #
    # or alternatively, just have a "UdpPacketizer" and "UdpSender/etc"
    # as separate modules! (the sender has an actual connection object)
    #
    # how to deal with multithreading/multiprocessing?
    # perhaps each module specifies which pool it wants to be in
    # (CPU/IO/SINGLE)
    #
    # OK, but what about messages? or model switching?
    # Who handles all that?
    #
    # So... what's the point of Executor?
    # Backpressure handling?
    # Auto-model switching?
    # "Outer" comm protocol? (which is used for... what?)
    #
    def __init__(self, model):
        self.model = model
