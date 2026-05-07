// This is a generated file - do not edit.
//
// Generated from imu.proto.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports
// ignore_for_file: unused_import

import 'dart:convert' as $convert;
import 'dart:core' as $core;
import 'dart:typed_data' as $typed_data;

@$core.Deprecated('Use iMUSampleDescriptor instead')
const IMUSample$json = {
  '1': 'IMUSample',
  '2': [
    {
      '1': 'relative_timestamp_ms',
      '3': 1,
      '4': 1,
      '5': 4,
      '10': 'relativeTimestampMs'
    },
    {'1': 'acc_x', '3': 2, '4': 1, '5': 2, '10': 'accX'},
    {'1': 'acc_y', '3': 3, '4': 1, '5': 2, '10': 'accY'},
    {'1': 'acc_z', '3': 4, '4': 1, '5': 2, '10': 'accZ'},
    {'1': 'gyro_x', '3': 5, '4': 1, '5': 2, '10': 'gyroX'},
    {'1': 'gyro_y', '3': 6, '4': 1, '5': 2, '10': 'gyroY'},
    {'1': 'gyro_z', '3': 7, '4': 1, '5': 2, '10': 'gyroZ'},
    {'1': 'quat_w', '3': 8, '4': 1, '5': 2, '10': 'quatW'},
    {'1': 'quat_x', '3': 9, '4': 1, '5': 2, '10': 'quatX'},
    {'1': 'quat_y', '3': 10, '4': 1, '5': 2, '10': 'quatY'},
    {'1': 'quat_z', '3': 11, '4': 1, '5': 2, '10': 'quatZ'},
  ],
};

/// Descriptor for `IMUSample`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List iMUSampleDescriptor = $convert.base64Decode(
    'CglJTVVTYW1wbGUSMgoVcmVsYXRpdmVfdGltZXN0YW1wX21zGAEgASgEUhNyZWxhdGl2ZVRpbW'
    'VzdGFtcE1zEhMKBWFjY194GAIgASgCUgRhY2NYEhMKBWFjY195GAMgASgCUgRhY2NZEhMKBWFj'
    'Y196GAQgASgCUgRhY2NaEhUKBmd5cm9feBgFIAEoAlIFZ3lyb1gSFQoGZ3lyb195GAYgASgCUg'
    'VneXJvWRIVCgZneXJvX3oYByABKAJSBWd5cm9aEhUKBnF1YXRfdxgIIAEoAlIFcXVhdFcSFQoG'
    'cXVhdF94GAkgASgCUgVxdWF0WBIVCgZxdWF0X3kYCiABKAJSBXF1YXRZEhUKBnF1YXRfehgLIA'
    'EoAlIFcXVhdFo=');

@$core.Deprecated('Use iMUStreamDescriptor instead')
const IMUStream$json = {
  '1': 'IMUStream',
  '2': [
    {
      '1': 'samples',
      '3': 1,
      '4': 3,
      '5': 11,
      '6': '.IMUSample',
      '10': 'samples'
    },
  ],
};

/// Descriptor for `IMUStream`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List iMUStreamDescriptor = $convert.base64Decode(
    'CglJTVVTdHJlYW0SJAoHc2FtcGxlcxgBIAMoCzIKLklNVVNhbXBsZVIHc2FtcGxlcw==');
