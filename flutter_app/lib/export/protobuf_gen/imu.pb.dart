// This is a generated file - do not edit.
//
// Generated from imu.proto.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports

import 'dart:core' as $core;

import 'package:fixnum/fixnum.dart' as $fixnum;
import 'package:protobuf/protobuf.dart' as $pb;

export 'package:protobuf/protobuf.dart' show GeneratedMessageGenericExtensions;

class IMUSample extends $pb.GeneratedMessage {
  factory IMUSample({
    $fixnum.Int64? relativeTimestampMs,
    $core.double? accX,
    $core.double? accY,
    $core.double? accZ,
    $core.double? gyroX,
    $core.double? gyroY,
    $core.double? gyroZ,
    $core.double? quatW,
    $core.double? quatX,
    $core.double? quatY,
    $core.double? quatZ,
  }) {
    final result = create();
    if (relativeTimestampMs != null)
      result.relativeTimestampMs = relativeTimestampMs;
    if (accX != null) result.accX = accX;
    if (accY != null) result.accY = accY;
    if (accZ != null) result.accZ = accZ;
    if (gyroX != null) result.gyroX = gyroX;
    if (gyroY != null) result.gyroY = gyroY;
    if (gyroZ != null) result.gyroZ = gyroZ;
    if (quatW != null) result.quatW = quatW;
    if (quatX != null) result.quatX = quatX;
    if (quatY != null) result.quatY = quatY;
    if (quatZ != null) result.quatZ = quatZ;
    return result;
  }

  IMUSample._();

  factory IMUSample.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory IMUSample.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'IMUSample',
      createEmptyInstance: create)
    ..a<$fixnum.Int64>(
        1, _omitFieldNames ? '' : 'relativeTimestampMs', $pb.PbFieldType.OU6,
        defaultOrMaker: $fixnum.Int64.ZERO)
    ..aD(2, _omitFieldNames ? '' : 'accX', fieldType: $pb.PbFieldType.OF)
    ..aD(3, _omitFieldNames ? '' : 'accY', fieldType: $pb.PbFieldType.OF)
    ..aD(4, _omitFieldNames ? '' : 'accZ', fieldType: $pb.PbFieldType.OF)
    ..aD(5, _omitFieldNames ? '' : 'gyroX', fieldType: $pb.PbFieldType.OF)
    ..aD(6, _omitFieldNames ? '' : 'gyroY', fieldType: $pb.PbFieldType.OF)
    ..aD(7, _omitFieldNames ? '' : 'gyroZ', fieldType: $pb.PbFieldType.OF)
    ..aD(8, _omitFieldNames ? '' : 'quatW', fieldType: $pb.PbFieldType.OF)
    ..aD(9, _omitFieldNames ? '' : 'quatX', fieldType: $pb.PbFieldType.OF)
    ..aD(10, _omitFieldNames ? '' : 'quatY', fieldType: $pb.PbFieldType.OF)
    ..aD(11, _omitFieldNames ? '' : 'quatZ', fieldType: $pb.PbFieldType.OF)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IMUSample clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IMUSample copyWith(void Function(IMUSample) updates) =>
      super.copyWith((message) => updates(message as IMUSample)) as IMUSample;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static IMUSample create() => IMUSample._();
  @$core.override
  IMUSample createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static IMUSample getDefault() =>
      _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<IMUSample>(create);
  static IMUSample? _defaultInstance;

  @$pb.TagNumber(1)
  $fixnum.Int64 get relativeTimestampMs => $_getI64(0);
  @$pb.TagNumber(1)
  set relativeTimestampMs($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasRelativeTimestampMs() => $_has(0);
  @$pb.TagNumber(1)
  void clearRelativeTimestampMs() => $_clearField(1);

  @$pb.TagNumber(2)
  $core.double get accX => $_getN(1);
  @$pb.TagNumber(2)
  set accX($core.double value) => $_setFloat(1, value);
  @$pb.TagNumber(2)
  $core.bool hasAccX() => $_has(1);
  @$pb.TagNumber(2)
  void clearAccX() => $_clearField(2);

  @$pb.TagNumber(3)
  $core.double get accY => $_getN(2);
  @$pb.TagNumber(3)
  set accY($core.double value) => $_setFloat(2, value);
  @$pb.TagNumber(3)
  $core.bool hasAccY() => $_has(2);
  @$pb.TagNumber(3)
  void clearAccY() => $_clearField(3);

  @$pb.TagNumber(4)
  $core.double get accZ => $_getN(3);
  @$pb.TagNumber(4)
  set accZ($core.double value) => $_setFloat(3, value);
  @$pb.TagNumber(4)
  $core.bool hasAccZ() => $_has(3);
  @$pb.TagNumber(4)
  void clearAccZ() => $_clearField(4);

  @$pb.TagNumber(5)
  $core.double get gyroX => $_getN(4);
  @$pb.TagNumber(5)
  set gyroX($core.double value) => $_setFloat(4, value);
  @$pb.TagNumber(5)
  $core.bool hasGyroX() => $_has(4);
  @$pb.TagNumber(5)
  void clearGyroX() => $_clearField(5);

  @$pb.TagNumber(6)
  $core.double get gyroY => $_getN(5);
  @$pb.TagNumber(6)
  set gyroY($core.double value) => $_setFloat(5, value);
  @$pb.TagNumber(6)
  $core.bool hasGyroY() => $_has(5);
  @$pb.TagNumber(6)
  void clearGyroY() => $_clearField(6);

  @$pb.TagNumber(7)
  $core.double get gyroZ => $_getN(6);
  @$pb.TagNumber(7)
  set gyroZ($core.double value) => $_setFloat(6, value);
  @$pb.TagNumber(7)
  $core.bool hasGyroZ() => $_has(6);
  @$pb.TagNumber(7)
  void clearGyroZ() => $_clearField(7);

  @$pb.TagNumber(8)
  $core.double get quatW => $_getN(7);
  @$pb.TagNumber(8)
  set quatW($core.double value) => $_setFloat(7, value);
  @$pb.TagNumber(8)
  $core.bool hasQuatW() => $_has(7);
  @$pb.TagNumber(8)
  void clearQuatW() => $_clearField(8);

  @$pb.TagNumber(9)
  $core.double get quatX => $_getN(8);
  @$pb.TagNumber(9)
  set quatX($core.double value) => $_setFloat(8, value);
  @$pb.TagNumber(9)
  $core.bool hasQuatX() => $_has(8);
  @$pb.TagNumber(9)
  void clearQuatX() => $_clearField(9);

  @$pb.TagNumber(10)
  $core.double get quatY => $_getN(9);
  @$pb.TagNumber(10)
  set quatY($core.double value) => $_setFloat(9, value);
  @$pb.TagNumber(10)
  $core.bool hasQuatY() => $_has(9);
  @$pb.TagNumber(10)
  void clearQuatY() => $_clearField(10);

  @$pb.TagNumber(11)
  $core.double get quatZ => $_getN(10);
  @$pb.TagNumber(11)
  set quatZ($core.double value) => $_setFloat(10, value);
  @$pb.TagNumber(11)
  $core.bool hasQuatZ() => $_has(10);
  @$pb.TagNumber(11)
  void clearQuatZ() => $_clearField(11);
}

class IMUStream extends $pb.GeneratedMessage {
  factory IMUStream({
    $core.Iterable<IMUSample>? samples,
  }) {
    final result = create();
    if (samples != null) result.samples.addAll(samples);
    return result;
  }

  IMUStream._();

  factory IMUStream.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory IMUStream.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'IMUStream',
      createEmptyInstance: create)
    ..pPM<IMUSample>(1, _omitFieldNames ? '' : 'samples',
        subBuilder: IMUSample.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IMUStream clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IMUStream copyWith(void Function(IMUStream) updates) =>
      super.copyWith((message) => updates(message as IMUStream)) as IMUStream;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static IMUStream create() => IMUStream._();
  @$core.override
  IMUStream createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static IMUStream getDefault() =>
      _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<IMUStream>(create);
  static IMUStream? _defaultInstance;

  @$pb.TagNumber(1)
  $pb.PbList<IMUSample> get samples => $_getList(0);
}

const $core.bool _omitFieldNames =
    $core.bool.fromEnvironment('protobuf.omit_field_names');
const $core.bool _omitMessageNames =
    $core.bool.fromEnvironment('protobuf.omit_message_names');
