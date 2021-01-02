from dataclasses import dataclass


@dataclass
class MuxPacket:
    stream_id: int
    payload_size: int
    payload: bytes

    def to_bytes(self):
        b_stream_id = self.stream_id.to_bytes(4, byteorder="big")
        b_payload_size = self.payload_size.to_bytes(4, byteorder="big")
        return b"".join([b_stream_id, b_payload_size, self.payload])


__all__ = [
    "MuxPacket",
]
