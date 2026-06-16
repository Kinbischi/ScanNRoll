import socket
import struct
import time
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional

UDP_IP = "192.168.0.251"
UDP_PORT = 1234  # set this to your configured sensor port
HDF5_FILE = "udp_profiles.h5"

#notes: optional out, frame index and type out of dataclass
# make it that everything fills class (no intermediate dicts)
# find_next_index out
# check in ProfileData class if stuff is there --> then only send package
# generate a reader function to read in hdf5
@dataclass
class MeasurementData:
    config_mode: bool
    time_synced: bool
    values_valid: bool
    alarm: bool
    quality: int
    output1: bool
    output2: bool
    measurements: tuple
    measurement_rate_hz: float
    timestamp_sec: int
    timestamp_usec: int
    encoderPosition: int

@dataclass
class ProfileData:
    block_id: int
    frame_type: int
    frame_index: int
    source_ip: str
    measurement: Optional[MeasurementData] = None
    x: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    time_synced: Optional[bool] = None
    values_valid: Optional[bool] = None
    quality: Optional[int] = None
    timestamp_sec: Optional[int] = None
    timestamp_usec: Optional[int] = None
    encoderValue: Optional[int] = None
    config_mode: Optional[bool] = None
    measurement_rate_hz: Optional[float] = None
    profile_length: Optional[int] = None
    measurement_block_id: Optional[int] = None
    arrival_time: Optional[float] = None

def parse_header(data: bytes):
    if len(data) < 8:
        raise ValueError("Packet too short")

    block_id, = struct.unpack_from("<I", data, 0)
    frame_type, = struct.unpack_from("<B", data, 4)
    frame_index, = struct.unpack_from("<H", data, 6) 
    return block_id, frame_type, frame_index

class HDF5ProfileWriter:
    def __init__(self, path: str):
        self.file = h5py.File(path, "a")
        self.next_index = self._find_next_index()

    def _find_next_index(self) -> int:
        names = [name for name in self.file.keys() if name.startswith("profile_")]
        if not names:
            return 0
        return max(int(name.split("_", 1)[1]) for name in names) + 1

    def write_profile(self, data: ProfileData):
        group_name = f"profile_{self.next_index:06d}"
        group = self.file.create_group(group_name)
        group.attrs["block_id"] = data.block_id
        group.attrs["frame_type"] = data.frame_type
        group.attrs["frame_index"] = data.frame_index
        group.attrs["source_ip"] = data.source_ip

        if data.measurement:
            for key, value in data.measurement.__dict__.items():
                if isinstance(value, (list, tuple)):
                    group.create_dataset(f"measurement_{key}", data=np.asarray(value), compression="gzip")
                else:
                    group.attrs[f"measurement_{key}"] = value

        # Write zProfile-specific fields
        if data.config_mode is not None:
            group.attrs["config_mode"] = data.config_mode
        if data.time_synced is not None:
            group.attrs["time_synced"] = data.time_synced
        if data.values_valid is not None:
            group.attrs["values_valid"] = data.values_valid
        if data.quality is not None:
            group.attrs["quality"] = data.quality
        if data.measurement_rate_hz is not None:
            group.attrs["measurement_rate_hz"] = data.measurement_rate_hz
        if data.timestamp_sec is not None:
            group.attrs["timestamp_sec"] = data.timestamp_sec
        if data.timestamp_usec is not None:
            group.attrs["timestamp_usec"] = data.timestamp_usec
        if data.encoderValue is not None:
            group.attrs["encoderValue"] = data.encoderValue
        if data.profile_length is not None:
            group.attrs["profile_length"] = data.profile_length
        if data.measurement_block_id is not None:
            group.attrs["measurement_block_id"] = data.measurement_block_id
        if data.arrival_time is not None:
            group.attrs["arrival_time"] = data.arrival_time

        if data.x is not None and data.z is not None:
            group.create_dataset("x", data=data.x, compression="gzip")
            group.create_dataset("z", data=data.z, compression="gzip")

        self.file.flush()
        self.next_index += 1

    def close(self):
        self.file.close()


def parse_udp_packet_zProfile(data: bytes) -> ProfileData:
    if len(data) < 32:
        raise ValueError("Z-profile packet too short")

    block_id, frame_type, frame_index = parse_header(data)
    body_offset = 8
    message_type, = struct.unpack_from("<B", data, body_offset)

    if message_type != 1:
        raise ValueError(f"Unexpected message_type for Z profile: {message_type}")

    # Parse additional fields from zProfile
    config_mode = struct.unpack_from("<?", data, body_offset + 1)[0]
    time_synced = struct.unpack_from("<?", data, body_offset + 2)[0]
    values_valid = struct.unpack_from("<?", data, body_offset + 3)[0]
    quality = struct.unpack_from("<B", data, body_offset + 5)[0]
    measurement_rate_hz = struct.unpack_from("<f", data, body_offset + 36)[0]
    timestamp_sec = struct.unpack_from("<I", data, body_offset + 10)[0]
    timestamp_usec = struct.unpack_from("<I", data, body_offset + 14)[0]
    encoderValue = struct.unpack_from("<H", data, body_offset + 18)[0]
    profile_length = struct.unpack_from("<I", data, body_offset + 20)[0]

    expected_end = body_offset + 24 + profile_length * 4
    if len(data) < expected_end:
        raise ValueError("Incomplete Z-profile payload")

    x = np.empty(profile_length, dtype=np.int16)
    z = np.empty(profile_length, dtype=np.uint16)

    for i in range(profile_length):
        offset = body_offset + 24 + 4 * i
        x[i] = struct.unpack_from("<h", data, offset)[0]
        z[i] = struct.unpack_from("<H", data, offset + 2)[0]

    return ProfileData(
        block_id=block_id,
        frame_type=frame_type,
        frame_index=frame_index,
        source_ip="",  # Will be set later
        measurement=None,  # Will be set later
        x=x,
        z=z,
        config_mode=config_mode,
        time_synced=time_synced,
        values_valid=values_valid,
        quality=quality,
        measurement_rate_hz=measurement_rate_hz,
        timestamp_sec=timestamp_sec,
        timestamp_usec=timestamp_usec,
        encoderValue=encoderValue,
        profile_length=profile_length,
        arrival_time=time.time()
    )


def parse_udp_packet_measure(data: bytes) -> MeasurementData:
    # -------------------------
    # HEADER (8 bytes)
    # -------------------------

    block_id, = struct.unpack_from("<I", data, 0)
    frame_type, = struct.unpack_from("<B", data, 4)
    frame_count_or_index, = struct.unpack_from("<H", data, 6)

    # Header not used in current implementation

    # -------------------------
    # BODY
    # -------------------------
    body_offset = 8

    message_type, = struct.unpack_from("<B", data, body_offset)

    # According to spec (offsets from body start) :contentReference[oaicite:1]{index=1}
    config_mode = struct.unpack_from("<?", data, body_offset + 1)[0]
    time_synced = struct.unpack_from("<?", data, body_offset + 2)[0]
    values_valid = struct.unpack_from("<?", data, body_offset + 3)[0]
    alarm = struct.unpack_from("<?", data, body_offset + 4)[0]
    quality = struct.unpack_from("<B", data, body_offset + 5)[0]
    output1 = struct.unpack_from("<?", data, body_offset + 6)[0]
    output2 = struct.unpack_from("<?", data, body_offset + 7)[0]

    # 7 float32 measurement values
    measurements = struct.unpack_from("<7f", data, body_offset + 8)

    measurement_rate, = struct.unpack_from("<f", data, body_offset + 36)
    timestamp_sec, = struct.unpack_from("<I", data, body_offset + 40)
    timestamp_usec, = struct.unpack_from("<I", data, body_offset + 44)
    encoderPosition, = struct.unpack_from("<H", data, body_offset + 48)

    return MeasurementData(
        config_mode=config_mode,
        time_synced=time_synced,
        values_valid=values_valid,
        alarm=alarm,
        quality=quality,
        output1=output1,
        output2=output2,
        measurements=measurements,
        measurement_rate_hz=measurement_rate,
        timestamp_sec=timestamp_sec,
        timestamp_usec=timestamp_usec,
        encoderPosition=encoderPosition
    )


def run_udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    writer = HDF5ProfileWriter(HDF5_FILE)
    measurement_dict = {}  # block_id -> MeasurementData
    zprofile_dict = {}  # block_id -> ProfileData
    last_z_fragment = None

    print(f"Listening on {UDP_IP}:{UDP_PORT}...")

    while True:
        data, addr = sock.recvfrom(65535)

        try:
            block_id, frame_type, frame_index = parse_header(data)
            body_offset = 8
            message_type, = struct.unpack_from("<B", data, body_offset)

            if frame_type == 0 and message_type == 0: #measurement packet
                measurement_dict[block_id] = parse_udp_packet_measure(data)
                print(f"Measurement packet received: block={block_id} frame={frame_index}")
                # Check if corresponding zProfile is waiting
                expected_z_block_id = block_id - 1
                if expected_z_block_id in zprofile_dict:
                    zprofile = zprofile_dict[expected_z_block_id]
                    zprofile.measurement = measurement_dict[block_id]
                    zprofile.measurement_block_id = block_id
                    writer.write_profile(zprofile)
                    print(f"Wrote profile {writer.next_index - 1} with {len(zprofile.x)} points")
                    del zprofile_dict[expected_z_block_id]
                    del measurement_dict[block_id]

            elif frame_type == 1 and message_type == 1: # start of Z profile block
                last_z_fragment = data # data including header
                print(f"Started Z profile block {block_id}")

            elif frame_type == 2 and last_z_fragment is not None:
                source_block_id = struct.unpack_from("<I", last_z_fragment, 0)[0]
                if block_id != source_block_id: # split z-profile packages have to come with same block id
                    print("Warning: Z profile continuation block_id mismatch")
                    
                zProfileData = b"".join([last_z_fragment, data[body_offset:]])
                profile_length = struct.unpack_from("<I", zProfileData, body_offset + 20)[0]
                total_length = len(zProfileData)

                if total_length - body_offset - 24 == profile_length * 4: # all profile points have arrived
                    profile_data = parse_udp_packet_zProfile(zProfileData)
                    profile_data.source_ip = addr[0]
                    expected_measurement_block_id = profile_data.block_id + 1
                    if expected_measurement_block_id in measurement_dict:
                        profile_data.measurement = measurement_dict[expected_measurement_block_id]
                        profile_data.measurement_block_id = expected_measurement_block_id
                        del measurement_dict[expected_measurement_block_id]
                        writer.write_profile(profile_data)
                        print(f"Wrote profile {writer.next_index - 1} with {len(profile_data.x)} points")
                    else:
                        # Store for later pairing
                        zprofile_dict[profile_data.block_id] = profile_data
                        print(f"Stored zProfile block {profile_data.block_id}, waiting for measurement block {expected_measurement_block_id}")
                    last_z_fragment = None
                else:
                    print("Waiting for more Z profile fragments")

        except Exception as e:
            print(f"Parse error: {e}")


if __name__ == "__main__":
    run_udp_listener()


"""
TODO:
NTP time measurement --> should i use this? when is it active (ip adress needed? in parameter config)
time sync via python? just over real times (from beckhoff and this python script --> synch by writing start time into file)
"""
