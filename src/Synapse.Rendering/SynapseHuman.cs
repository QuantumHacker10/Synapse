// =============================================================================
// SynapseHuman.cs - GDNN Engine: Production Artist Pipeline
// Digital Human Character System - MetaHuman equivalent
// Complete skeletal mesh, morph targets, skin SSS, hair, eyes, teeth,
// clothing, LOD, and procedural body/face generation.
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Materials.SubstrateOmega;
using GDNN.Rendering.MeshIO;

namespace GDNN.Rendering.ArtPipeline
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    public enum HumanBodyRegion : byte
    {
        Head, Neck, Torso, UpperArm, LowerArm, Hand,
        UpperLeg, LowerLeg, Foot, FullBody
    }

    public enum HumanSkinTone : byte
    {
        Porcelain, Fair, Light, Medium, Olive, Tan,
        Brown, Dark, Ebony, Custom
    }

    public enum HumanHairStyle : byte
    {
        Short, Medium, Long, Curly, Afro, Braids,
        Ponytail, Bun, BuzzCut, Bald, Beard, Mustache,
        Goatee, Sideburns, Custom
    }

    public enum HumanEyeColor : byte
    {
        Blue, Green, Brown, Hazel, Grey, Amber,
        Violet, Heterochromia, Custom
    }

    public enum HumanEthnicity : byte
    {
        European, EastAsian, SouthAsian, MiddleEastern,
        African, NativeAmerican, PacificIslander, Mixed, Custom
    }

    public enum HumanGender : byte
    {
        Male, Female, NonBinary, Custom
    }

    public enum HumanAgeGroup : byte
    {
        Infant, Child, Teenager, YoungAdult, Adult, MiddleAged, Elderly
    }

    public enum HumanLODLevel : byte
    {
        LOD0_FullDetail,
        LOD1_High,
        LOD2_Medium,
        LOD3_Low,
        LOD4_Billboard
    }

    public enum ClothingType : byte
    {
        Shirt, Pants, Dress, Skirt, Jacket, Coat,
        Shoes, Hat, Gloves, Scarf, Belt, Armor,
        Underwear, Suit, Uniform, Casual, Custom
    }

    public enum ClothingMaterial : byte
    {
        Cotton, Silk, Polyester, Leather, Denim,
        Wool, Nylon, Spandex, Metal, Kevlar,
        Rubber, Latex, Velvet, Cashmere, Custom
    }

    // =========================================================================
    // MORPH TARGET / BLEND SHAPE
    // =========================================================================

    public class MorphTarget
    {
        public string Name { get; set; } = "";
        public string Category { get; set; } = "";
        public float MinValue { get; set; } = -1.0f;
        public float MaxValue { get; set; } = 1.0f;
        public float DefaultValue { get; set; }
        public float CurrentValue { get; set; }
        public Vector3[] DeltaPositions { get; set; } = Array.Empty<Vector3>();
        public Vector3[] DeltaNormals { get; set; } = Array.Empty<Vector3>();
        public Vector4[] DeltaTangents { get; set; } = Array.Empty<Vector4>();
        public bool IsInverted { get; set; }
        public string LinkedTarget { get; set; } = "";
        public float LinkStrength { get; set; } = 1.0f;

        public float Apply(float weight)
        {
            float clamped = Math.Clamp(weight, MinValue, MaxValue);
            CurrentValue = clamped;
            return clamped;
        }

        public static MorphTarget CreateEmpty(string name) => new() { Name = name };
    }

    // =========================================================================
    // MORPH TARGET GROUP
    // =========================================================================

    public class MorphTargetGroup
    {
        public string Name { get; set; } = "";
        public string DisplayName { get; set; } = "";
        public List<MorphTarget> Targets { get; set; } = new();
        public bool IsVisible { get; set; } = true;
        public int SortOrder { get; set; }

        public void SetWeight(string targetName, float weight)
        {
            var target = Targets.FirstOrDefault(t => t.Name == targetName);
            target?.Apply(weight);
        }

        public float GetWeight(string targetName)
        {
            var target = Targets.FirstOrDefault(t => t.Name == targetName);
            return target?.CurrentValue ?? 0f;
        }

        public void ResetAll()
        {
            foreach (var target in Targets)
                target.CurrentValue = target.DefaultValue;
        }

        public static MorphTargetGroup CreateFaceExpressions()
        {
            return new MorphTargetGroup
            {
                Name = "FaceExpressions",
                DisplayName = "Facial Expressions",
                Targets = new List<MorphTarget>
                {
                    new() { Name = "BrowInnerUp", Category = "Brows", MinValue = 0, MaxValue = 1 },
                    new() { Name = "BrowOuterUpLeft", Category = "Brows", MinValue = 0, MaxValue = 1 },
                    new() { Name = "BrowOuterUpRight", Category = "Brows", MinValue = 0, MaxValue = 1 },
                    new() { Name = "BrowDownLeft", Category = "Brows", MinValue = 0, MaxValue = 1 },
                    new() { Name = "BrowDownRight", Category = "Brows", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeSquintLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeSquintRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeWideLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeWideRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeBlinkLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeBlinkRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookDownLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookDownRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookUpLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookUpRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookInLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookInRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookOutLeft", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "EyeLookOutRight", Category = "Eyes", MinValue = 0, MaxValue = 1 },
                    new() { Name = "CheekPuff", Category = "Cheeks", MinValue = 0, MaxValue = 1 },
                    new() { Name = "CheekSquintLeft", Category = "Cheeks", MinValue = 0, MaxValue = 1 },
                    new() { Name = "CheekSquintRight", Category = "Cheeks", MinValue = 0, MaxValue = 1 },
                    new() { Name = "NoseSneerLeft", Category = "Nose", MinValue = 0, MaxValue = 1 },
                    new() { Name = "NoseSneerRight", Category = "Nose", MinValue = 0, MaxValue = 1 },
                    new() { Name = "NoseScrunch", Category = "Nose", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthSmileLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthSmileRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthFrownLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthFrownRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthDimpleLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthDimpleRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthClose", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthFunnel", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthPucker", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthRollUpper", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthRollLower", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthShrugLower", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthShrugUpper", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthPressLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthPressRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthLowerDownLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthLowerDownRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthUpperUpLeft", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "MouthUpperUpRight", Category = "Mouth", MinValue = 0, MaxValue = 1 },
                    new() { Name = "TongueOut", Category = "Tongue", MinValue = 0, MaxValue = 1 },
                    new() { Name = "TongueUp", Category = "Tongue", MinValue = 0, MaxValue = 1 },
                    new() { Name = "TongueDown", Category = "Tongue", MinValue = 0, MaxValue = 1 },
                    new() { Name = "TongueLeft", Category = "Tongue", MinValue = 0, MaxValue = 1 },
                    new() { Name = "TongueRight", Category = "Tongue", MinValue = 0, MaxValue = 1 },
                    new() { Name = "JawForward", Category = "Jaw", MinValue = 0, MaxValue = 1 },
                    new() { Name = "JawLeft", Category = "Jaw", MinValue = 0, MaxValue = 1 },
                    new() { Name = "JawRight", Category = "Jaw", MinValue = 0, MaxValue = 1 },
                    new() { Name = "JawOpen", Category = "Jaw", MinValue = 0, MaxValue = 1 },
                }
            };
        }

        public static MorphTargetGroup CreateFaceShape()
        {
            return new MorphTargetGroup
            {
                Name = "FaceShape",
                DisplayName = "Face Shape",
                Targets = new List<MorphTarget>
                {
                    new() { Name = "HeadScale", Category = "Head", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "HeadWidth", Category = "Head", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "HeadLength", Category = "Head", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "JawWidth", Category = "Jaw", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "JawHeight", Category = "Jaw", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ChinWidth", Category = "Chin", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "ChinHeight", Category = "Chin", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ChinForward", Category = "Chin", MinValue = -1, MaxValue = 1, DefaultValue = 0 },
                    new() { Name = "CheekboneWidth", Category = "Cheeks", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "CheekboneHeight", Category = "Cheeks", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "CheekFullness", Category = "Cheeks", MinValue = 0.5f, MaxValue = 1.5f, DefaultValue = 1.0f },
                    new() { Name = "NoseWidth", Category = "Nose", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "NoseHeight", Category = "Nose", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "NoseBridge", Category = "Nose", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "NoseTip", Category = "Nose", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "NoseBulge", Category = "Nose", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "EyeSize", Category = "Eyes", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "EyeSpacing", Category = "Eyes", MinValue = 0.9f, MaxValue = 1.1f, DefaultValue = 1.0f },
                    new() { Name = "EyeHeight", Category = "Eyes", MinValue = 0.9f, MaxValue = 1.1f, DefaultValue = 1.0f },
                    new() { Name = "EyeAngle", Category = "Eyes", MinValue = -1, MaxValue = 1, DefaultValue = 0 },
                    new() { Name = "LipWidth", Category = "Mouth", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "LipHeight", Category = "Mouth", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "UpperLipThickness", Category = "Mouth", MinValue = 0.5f, MaxValue = 1.5f, DefaultValue = 1.0f },
                    new() { Name = "LowerLipThickness", Category = "Mouth", MinValue = 0.5f, MaxValue = 1.5f, DefaultValue = 1.0f },
                    new() { Name = "MouthWidth", Category = "Mouth", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "EarSize", Category = "Ears", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "EarAngle", Category = "Ears", MinValue = -1, MaxValue = 1, DefaultValue = 0 },
                    new() { Name = "NeckWidth", Category = "Neck", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "NeckLength", Category = "Neck", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                }
            };
        }

        public static MorphTargetGroup CreateBodyShape()
        {
            return new MorphTargetGroup
            {
                Name = "BodyShape",
                DisplayName = "Body Shape",
                Targets = new List<MorphTarget>
                {
                    new() { Name = "Height", Category = "Overall", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "Weight", Category = "Overall", MinValue = 0.5f, MaxValue = 1.5f, DefaultValue = 1.0f },
                    new() { Name = "Muscularity", Category = "Overall", MinValue = 0.5f, MaxValue = 1.5f, DefaultValue = 1.0f },
                    new() { Name = "ShoulderWidth", Category = "Torso", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ChestSize", Category = "Torso", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "WaistWidth", Category = "Torso", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "HipWidth", Category = "Torso", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ArmLength", Category = "Arms", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ArmThickness", Category = "Arms", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "ForearmThickness", Category = "Arms", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "HandSize", Category = "Hands", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "FingerLength", Category = "Hands", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "LegLength", Category = "Legs", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "ThighThickness", Category = "Legs", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "CalfThickness", Category = "Legs", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "FootSize", Category = "Feet", MinValue = 0.8f, MaxValue = 1.2f, DefaultValue = 1.0f },
                    new() { Name = "GluteSize", Category = "Torso", MinValue = 0.7f, MaxValue = 1.3f, DefaultValue = 1.0f },
                    new() { Name = "BellySize", Category = "Torso", MinValue = 0.7f, MaxValue = 1.5f, DefaultValue = 1.0f },
                }
            };
        }
    }

    // =========================================================================
    // SKELETON / BONE SYSTEM
    // =========================================================================

    public class HumanBone
    {
        public int Index { get; set; }
        public string Name { get; set; } = "";
        public int ParentIndex { get; set; } = -1;
        public Vector3 LocalPosition { get; set; }
        public Quaternion LocalRotation { get; set; } = Quaternion.Identity;
        public Vector3 LocalScale { get; set; } = Vector3.One;
        public Vector3 RestPosition { get; set; }
        public Quaternion RestRotation { get; set; } = Quaternion.Identity;
        public HumanBodyRegion Region { get; set; }
        public Matrix4x4 InverseBindMatrix { get; set; } = Matrix4x4.Identity;
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;
    }

    public class HumanSkeleton
    {
        public string Name { get; set; } = "HumanRig";
        public HumanBone[] Bones { get; set; } = Array.Empty<HumanBone>();
        public int BoneCount => Bones?.Length ?? 0;

        public static HumanSkeleton CreateDefaultSkeleton()
        {
            var bones = new List<HumanBone>();
            int idx = 0;

            bones.Add(new HumanBone { Index = idx, Name = "Root", ParentIndex = -1, LocalPosition = Vector3.Zero, Region = HumanBodyRegion.FullBody }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Hips", ParentIndex = 0, LocalPosition = new Vector3(0, 1.0f, 0), Region = HumanBodyRegion.Torso }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Spine", ParentIndex = 1, LocalPosition = new Vector3(0, 0.12f, 0), Region = HumanBodyRegion.Torso }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Spine1", ParentIndex = 2, LocalPosition = new Vector3(0, 0.12f, 0), Region = HumanBodyRegion.Torso }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Spine2", ParentIndex = 3, LocalPosition = new Vector3(0, 0.12f, 0), Region = HumanBodyRegion.Torso }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Neck", ParentIndex = 4, LocalPosition = new Vector3(0, 0.15f, 0), Region = HumanBodyRegion.Neck }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "Head", ParentIndex = 5, LocalPosition = new Vector3(0, 0.12f, 0), Region = HumanBodyRegion.Head }); idx++;

            bones.Add(new HumanBone { Index = idx, Name = "LeftShoulder", ParentIndex = 4, LocalPosition = new Vector3(0.08f, 0.06f, 0), Region = HumanBodyRegion.UpperArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftUpperArm", ParentIndex = 7, LocalPosition = new Vector3(0.12f, 0, 0), Region = HumanBodyRegion.UpperArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftLowerArm", ParentIndex = 8, LocalPosition = new Vector3(0.28f, 0, 0), Region = HumanBodyRegion.LowerArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftHand", ParentIndex = 9, LocalPosition = new Vector3(0.25f, 0, 0), Region = HumanBodyRegion.Hand }); idx++;

            bones.Add(new HumanBone { Index = idx, Name = "RightShoulder", ParentIndex = 4, LocalPosition = new Vector3(-0.08f, 0.06f, 0), Region = HumanBodyRegion.UpperArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightUpperArm", ParentIndex = 11, LocalPosition = new Vector3(-0.12f, 0, 0), Region = HumanBodyRegion.UpperArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightLowerArm", ParentIndex = 12, LocalPosition = new Vector3(-0.28f, 0, 0), Region = HumanBodyRegion.LowerArm }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightHand", ParentIndex = 13, LocalPosition = new Vector3(-0.25f, 0, 0), Region = HumanBodyRegion.Hand }); idx++;

            bones.Add(new HumanBone { Index = idx, Name = "LeftUpperLeg", ParentIndex = 1, LocalPosition = new Vector3(0.1f, 0, 0), Region = HumanBodyRegion.UpperLeg }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftLowerLeg", ParentIndex = 15, LocalPosition = new Vector3(0, -0.42f, 0), Region = HumanBodyRegion.LowerLeg }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftFoot", ParentIndex = 16, LocalPosition = new Vector3(0, -0.42f, 0), Region = HumanBodyRegion.Foot }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "LeftToes", ParentIndex = 17, LocalPosition = new Vector3(0, 0, 0.12f), Region = HumanBodyRegion.Foot }); idx++;

            bones.Add(new HumanBone { Index = idx, Name = "RightUpperLeg", ParentIndex = 1, LocalPosition = new Vector3(-0.1f, 0, 0), Region = HumanBodyRegion.UpperLeg }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightLowerLeg", ParentIndex = 19, LocalPosition = new Vector3(0, -0.42f, 0), Region = HumanBodyRegion.LowerLeg }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightFoot", ParentIndex = 20, LocalPosition = new Vector3(0, -0.42f, 0), Region = HumanBodyRegion.Foot }); idx++;
            bones.Add(new HumanBone { Index = idx, Name = "RightToes", ParentIndex = 21, LocalPosition = new Vector3(0, 0, 0.12f), Region = HumanBodyRegion.Foot }); idx++;

            return new HumanSkeleton { Bones = bones.ToArray() };
        }

        public HumanBone GetBone(string name) => Bones?.FirstOrDefault(b => b.Name == name);
        public HumanBone GetBone(int index) => (Bones != null && index >= 0 && index < Bones.Length) ? Bones[index] : null;
        public int GetBoneIndex(string name) => Array.FindIndex(Bones, b => b.Name == name);
    }

    // =========================================================================
    // ANIMATION CLIP
    // =========================================================================

    public class HumanAnimationClip
    {
        public string Name { get; set; } = "";
        public float Duration { get; set; }
        public float FrameRate { get; set; } = 30f;
        public bool Loop { get; set; }
        public AnimationChannel[] Channels { get; set; } = Array.Empty<AnimationChannel>();
    }

    public class AnimationChannel
    {
        public string BoneName { get; set; } = "";
        public Vector3Key[] PositionKeys { get; set; } = Array.Empty<Vector3Key>();
        public QuaternionKey[] RotationKeys { get; set; } = Array.Empty<QuaternionKey>();
        public Vector3Key[] ScaleKeys { get; set; } = Array.Empty<Vector3Key>();
    }

    public struct Vector3Key
    {
        public float Time;
        public Vector3 Value;
    }

    public struct QuaternionKey
    {
        public float Time;
        public Quaternion Value;
    }

    // =========================================================================
    // CLOTHING
    // =========================================================================

    public class HumanClothing
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N");
        public string Name { get; set; } = "";
        public ClothingType Type { get; set; }
        public ClothingMaterial Material { get; set; }
        public string MeshPath { get; set; } = "";
        public MeshAsset Mesh { get; set; }
        public SubstrateMaterial MaterialInstance { get; set; }
        public Color3 PrimaryColor { get; set; } = new Color3(0.3f, 0.3f, 0.35f);
        public Color3 SecondaryColor { get; set; } = Color3.Black;
        public float Roughness { get; set; } = 0.7f;
        public float Metallic { get; set; }
        public float Sheen { get; set; }
        public float SheenRoughness { get; set; } = 0.5f;
        public bool IsVisible { get; set; } = true;
        public bool CastShadows { get; set; } = true;
        public float Thickness { get; set; } = 0.5f;
        public string[] TexturePaths { get; set; } = Array.Empty<string>();

        public SubstrateMaterial ToMaterial()
        {
            var mat = new SubstrateMaterial(Name)
            {
                Domain = MaterialDomain.Surface,
                FeatureFlags = MaterialFeatureFlags.None
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, PrimaryColor, 0, 1, 0.8f));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, Roughness, 0, 1, 0.5f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, Metallic, 0, 1, 0f));
            mat.SetProperty("Sheen", new MaterialProperty("Sheen", MaterialPropertyType.Float, Sheen, 0, 1, 0f));
            mat.SetProperty("SheenRoughness", new MaterialProperty("SheenRoughness", MaterialPropertyType.Float, SheenRoughness, 0, 1, 0.5f));
            mat.SetProperty("Thickness", new MaterialProperty("Thickness", MaterialPropertyType.Float, Thickness, 0, 10, 0.5f));
            mat.ComputeActiveFeatures();
            MaterialInstance = mat;
            return mat;
        }
    }

    // =========================================================================
    // HUMAN LOD CONFIG
    // =========================================================================

    public class HumanLODConfig
    {
        public HumanLODLevel Level { get; set; }
        public float ScreenSize { get; set; }
        public int MaxMorphTargets { get; set; } = 150;
        public bool EnableMorphNormals { get; set; } = true;
        public bool EnableClothSimulation { get; set; } = true;
        public int MaxBonesPerVertex { get; set; } = 4;
        public string MeshPath { get; set; } = "";
        public float TriangleReduction { get; set; } = 1.0f;
    }

    // =========================================================================
    // SYNAPSE HUMAN - MAIN DIGITAL HUMAN SYSTEM
    // =========================================================================

    public class SynapseHuman : IDisposable
    {
        private string _id;
        private string _name;
        private HumanGender _gender;
        private HumanEthnicity _ethnicity;
        private HumanAgeGroup _ageGroup;
        private HumanSkinTone _skinTone;
        private HumanEyeColor _eyeColor;
        private HumanHairStyle _hairStyle;

        private HumanSkeleton _skeleton;
        private MorphTargetGroup _faceExpressions;
        private MorphTargetGroup _faceShape;
        private MorphTargetGroup _bodyShape;

        private SubstrateMaterial _skinMaterial;
        private SubstrateMaterial _eyeMaterial;
        private SubstrateMaterial _teethMaterial;
        private SubstrateMaterial _hairMaterial;
        private SubstrateMaterial _nailMaterial;
        private SubstrateMaterial _scleraMaterial;

        private HumanLODConfig[] _lodConfigs;
        private Dictionary<HumanLODLevel, string> _meshPaths;

        private List<HumanClothing> _clothing;
        private List<HumanAnimationClip> _animations;

        private MeshAsset _baseMesh;
        private MeshAsset _faceMesh;
        private MeshAsset _bodyMesh;

        private bool _disposed;

        public string Id => _id;
        public string Name { get => _name; set => _name = value; }
        public HumanGender Gender => _gender;
        public HumanEthnicity Ethnicity => _ethnicity;
        public HumanAgeGroup AgeGroup => _ageGroup;
        public HumanSkinTone SkinTone => _skinTone;
        public HumanEyeColor EyeColor => _eyeColor;
        public HumanHairStyle HairStyle => _hairStyle;
        public HumanSkeleton Skeleton => _skeleton;
        public MorphTargetGroup FaceExpressions => _faceExpressions;
        public MorphTargetGroup FaceShape => _faceShape;
        public MorphTargetGroup BodyShape => _bodyShape;
        public SubstrateMaterial SkinMaterial => _skinMaterial;
        public SubstrateMaterial EyeMaterial => _eyeMaterial;
        public SubstrateMaterial HairMaterial => _hairMaterial;
        public IReadOnlyList<HumanClothing> Clothing => _clothing.AsReadOnly();
        public IReadOnlyList<HumanAnimationClip> Animations => _animations.AsReadOnly();

        public SynapseHuman(string name = "SynapseHuman")
        {
            _id = Guid.NewGuid().ToString("N");
            _name = name;
            _gender = HumanGender.Female;
            _ethnicity = HumanEthnicity.European;
            _ageGroup = HumanAgeGroup.YoungAdult;
            _skinTone = HumanSkinTone.Fair;
            _eyeColor = HumanEyeColor.Blue;
            _hairStyle = HumanHairStyle.Medium;

            _skeleton = HumanSkeleton.CreateDefaultSkeleton();
            _faceExpressions = MorphTargetGroup.CreateFaceExpressions();
            _faceShape = MorphTargetGroup.CreateFaceShape();
            _bodyShape = MorphTargetGroup.CreateBodyShape();

            _clothing = new List<HumanClothing>();
            _animations = new List<HumanAnimationClip>();
            _meshPaths = new Dictionary<HumanLODLevel, string>();
            _lodConfigs = CreateDefaultLODConfigs();

            InitializeMaterials();
        }

        private void InitializeMaterials()
        {
            _skinMaterial = CreateSkinMaterial(_skinTone);
            _eyeMaterial = CreateEyeMaterial(_eyeColor);
            _scleraMaterial = CreateScleraMaterial();
            _teethMaterial = CreateTeethMaterial();
            _hairMaterial = CreateHairMaterial(_hairStyle);
            _nailMaterial = CreateNailMaterial();
        }

        public void SetAppearance(
            HumanSkinTone? skinTone = null,
            HumanEyeColor? eyeColor = null,
            HumanHairStyle? hairStyle = null,
            HumanGender? gender = null,
            HumanEthnicity? ethnicity = null,
            HumanAgeGroup? ageGroup = null)
        {
            if (skinTone.HasValue) { _skinTone = skinTone.Value; _skinMaterial = CreateSkinMaterial(_skinTone); }
            if (eyeColor.HasValue) { _eyeColor = eyeColor.Value; _eyeMaterial = CreateEyeMaterial(_eyeColor); }
            if (hairStyle.HasValue) { _hairStyle = hairStyle.Value; _hairMaterial = CreateHairMaterial(_hairStyle); }
            if (gender.HasValue) _gender = gender.Value;
            if (ethnicity.HasValue) _ethnicity = ethnicity.Value;
            if (ageGroup.HasValue) _ageGroup = ageGroup.Value;
        }

        public void SetFaceMorph(string morphName, float weight)
        {
            _faceExpressions.SetWeight(morphName, weight);
            var shapeTarget = _faceShape.Targets.FirstOrDefault(t => t.Name == morphName);
            shapeTarget?.Apply(weight);
        }

        public void SetBodyMorph(string morphName, float weight)
        {
            _bodyShape.SetWeight(morphName, weight);
        }

        public float GetFaceMorph(string morphName)
        {
            return _faceExpressions.GetWeight(morphName);
        }

        public void ResetFace()
        {
            _faceExpressions.ResetAll();
        }

        public void ResetBody()
        {
            _bodyShape.ResetAll();
        }

        public void AddClothing(HumanClothing clothing)
        {
            if (clothing.MaterialInstance == null)
                clothing.ToMaterial();
            _clothing.Add(clothing);
        }

        public bool RemoveClothing(string clothingId)
        {
            return _clothing.RemoveAll(c => c.Id == clothingId) > 0;
        }

        public HumanClothing GetClothing(ClothingType type)
        {
            return _clothing.FirstOrDefault(c => c.Type == type);
        }

        public void AddAnimation(HumanAnimationClip clip)
        {
            _animations.Add(clip);
        }

        public HumanAnimationClip GetAnimation(string name)
        {
            return _animations.FirstOrDefault(a => a.Name == name);
        }

        private SubstrateMaterial CreateSkinMaterial(HumanSkinTone tone)
        {
            var mat = new SubstrateMaterial($"Skin_{tone}")
            {
                Domain = MaterialDomain.Surface,
                FeatureFlags = MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.SubsurfaceProfile
            };
            mat.InitializeDefaults();

            Color3 baseColor = tone switch
            {
                HumanSkinTone.Porcelain => new Color3(0.95f, 0.85f, 0.8f),
                HumanSkinTone.Fair => new Color3(0.9f, 0.72f, 0.62f),
                HumanSkinTone.Light => new Color3(0.82f, 0.65f, 0.52f),
                HumanSkinTone.Medium => new Color3(0.75f, 0.55f, 0.42f),
                HumanSkinTone.Olive => new Color3(0.7f, 0.55f, 0.38f),
                HumanSkinTone.Tan => new Color3(0.65f, 0.48f, 0.35f),
                HumanSkinTone.Brown => new Color3(0.5f, 0.35f, 0.25f),
                HumanSkinTone.Dark => new Color3(0.45f, 0.3f, 0.22f),
                HumanSkinTone.Ebony => new Color3(0.3f, 0.2f, 0.15f),
                _ => new Color3(0.8f, 0.65f, 0.55f)
            };

            Color3 sssColor = tone switch
            {
                HumanSkinTone.Porcelain => new Color3(0.95f, 0.4f, 0.3f),
                HumanSkinTone.Fair => new Color3(0.9f, 0.3f, 0.2f),
                HumanSkinTone.Light => new Color3(0.85f, 0.3f, 0.18f),
                HumanSkinTone.Medium => new Color3(0.8f, 0.25f, 0.15f),
                HumanSkinTone.Olive => new Color3(0.75f, 0.25f, 0.12f),
                HumanSkinTone.Tan => new Color3(0.7f, 0.22f, 0.1f),
                HumanSkinTone.Brown => new Color3(0.6f, 0.2f, 0.1f),
                HumanSkinTone.Dark => new Color3(0.55f, 0.18f, 0.08f),
                HumanSkinTone.Ebony => new Color3(0.4f, 0.15f, 0.05f),
                _ => new Color3(0.8f, 0.3f, 0.2f)
            };

            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, baseColor));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.55f, 0, 1, 0.55f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f));
            mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(1.0f, 0.3f, 0.15f), 0, 5, 1f));
            mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, sssColor));
            mat.SetProperty("Specular", new MaterialProperty("Specular", MaterialPropertyType.Float, 0.5f, 0, 1, 0.5f));
            mat.SetProperty("NormalStrength", new MaterialProperty("NormalStrength", MaterialPropertyType.Float, 1.0f, 0, 2, 1f));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private SubstrateMaterial CreateEyeMaterial(HumanEyeColor eyeColor)
        {
            var mat = new SubstrateMaterial($"Eye_{eyeColor}")
            {
                Domain = MaterialDomain.Eye,
                FeatureFlags = MaterialFeatureFlags.EyeBRDF | MaterialFeatureFlags.EyeIrisRefraction
            };
            mat.InitializeDefaults();

            Color3 irisColor = eyeColor switch
            {
                HumanEyeColor.Blue => new Color3(0.3f, 0.5f, 0.8f),
                HumanEyeColor.Green => new Color3(0.35f, 0.55f, 0.3f),
                HumanEyeColor.Brown => new Color3(0.45f, 0.28f, 0.12f),
                HumanEyeColor.Hazel => new Color3(0.55f, 0.42f, 0.2f),
                HumanEyeColor.Grey => new Color3(0.55f, 0.57f, 0.6f),
                HumanEyeColor.Amber => new Color3(0.75f, 0.55f, 0.15f),
                HumanEyeColor.Violet => new Color3(0.5f, 0.25f, 0.65f),
                _ => new Color3(0.4f, 0.35f, 0.25f)
            };

            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, irisColor));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.05f, 0, 1, 0.05f));
            mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.376f, 1, 2.5f, 1.376f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private SubstrateMaterial CreateScleraMaterial()
        {
            var mat = new SubstrateMaterial("Sclera")
            {
                Domain = MaterialDomain.Eye,
                FeatureFlags = MaterialFeatureFlags.EyeBRDF
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, new Color3(0.95f, 0.93f, 0.88f)));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.02f, 0, 1, 0.02f));
            mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.376f, 1, 2.5f, 1.376f));
            mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(0.8f, 0.15f, 0.1f)));
            mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.85f, 0.2f, 0.15f)));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private SubstrateMaterial CreateTeethMaterial()
        {
            var mat = new SubstrateMaterial("Teeth")
            {
                Domain = MaterialDomain.Surface,
                FeatureFlags = MaterialFeatureFlags.ClearCoat
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, new Color3(0.92f, 0.9f, 0.82f)));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.15f, 0, 1, 0.15f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f));
            mat.SetProperty("ClearCoat", new MaterialProperty("ClearCoat", MaterialPropertyType.Float, 0.8f, 0, 1, 0f));
            mat.SetProperty("ClearCoatRoughness", new MaterialProperty("ClearCoatRoughness", MaterialPropertyType.Float, 0.05f, 0, 1, 0.01f));
            mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.63f, 1, 2.5f, 1.63f));
            mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(0.5f, 0.3f, 0.2f)));
            mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.85f, 0.6f, 0.5f)));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private SubstrateMaterial CreateHairMaterial(HumanHairStyle style)
        {
            var mat = new SubstrateMaterial($"Hair_{style}")
            {
                Domain = MaterialDomain.Hair,
                FeatureFlags = MaterialFeatureFlags.HairBRDF
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, new Color3(0.35f, 0.22f, 0.12f)));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.38f, 0, 1, 0.38f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f));
            mat.SetProperty("Sheen", new MaterialProperty("Sheen", MaterialPropertyType.Float, 0.2f, 0, 1, 0f));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private SubstrateMaterial CreateNailMaterial()
        {
            var mat = new SubstrateMaterial("Nails")
            {
                Domain = MaterialDomain.Surface,
                FeatureFlags = MaterialFeatureFlags.ClearCoat
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, new Color3(0.85f, 0.75f, 0.72f)));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.15f, 0, 1, 0.15f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f));
            mat.SetProperty("ClearCoat", new MaterialProperty("ClearCoat", MaterialPropertyType.Float, 0.9f, 0, 1, 0f));
            mat.SetProperty("ClearCoatRoughness", new MaterialProperty("ClearCoatRoughness", MaterialPropertyType.Float, 0.05f, 0, 1, 0.01f));
            mat.ComputeActiveFeatures();
            return mat;
        }

        private HumanLODConfig[] CreateDefaultLODConfigs()
        {
            return new[]
            {
                new HumanLODConfig { Level = HumanLODLevel.LOD0_FullDetail, ScreenSize = 1.0f, MaxMorphTargets = 150, EnableMorphNormals = true, EnableClothSimulation = true, MaxBonesPerVertex = 4, TriangleReduction = 1.0f },
                new HumanLODConfig { Level = HumanLODLevel.LOD1_High, ScreenSize = 0.5f, MaxMorphTargets = 80, EnableMorphNormals = true, EnableClothSimulation = true, MaxBonesPerVertex = 4, TriangleReduction = 0.75f },
                new HumanLODConfig { Level = HumanLODLevel.LOD2_Medium, ScreenSize = 0.25f, MaxMorphTargets = 40, EnableMorphNormals = false, EnableClothSimulation = false, MaxBonesPerVertex = 4, TriangleReduction = 0.5f },
                new HumanLODConfig { Level = HumanLODLevel.LOD3_Low, ScreenSize = 0.1f, MaxMorphTargets = 10, EnableMorphNormals = false, EnableClothSimulation = false, MaxBonesPerVertex = 2, TriangleReduction = 0.25f },
                new HumanLODConfig { Level = HumanLODLevel.LOD4_Billboard, ScreenSize = 0.02f, MaxMorphTargets = 0, EnableMorphNormals = false, EnableClothSimulation = false, MaxBonesPerVertex = 0, TriangleReduction = 0.0f },
            };
        }

        public void LoadFromDirectory(string directory)
        {
            if (!Directory.Exists(directory)) return;

            var metadataPath = Path.Combine(directory, "human.json");
            if (File.Exists(metadataPath))
            {
                var json = File.ReadAllText(metadataPath);
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                var data = JsonSerializer.Deserialize<HumanExportData>(json, options);
                if (data != null)
                {
                    _name = data.Name ?? _name;
                    _gender = data.Gender;
                    _ethnicity = data.Ethnicity;
                    _ageGroup = data.AgeGroup;
                    _skinTone = data.SkinTone;
                    _eyeColor = data.EyeColor;
                    _hairStyle = data.HairStyle;
                    SetAppearance(_skinTone, _eyeColor, _hairStyle, _gender, _ethnicity, _ageGroup);
                }
            }

            var meshFiles = Directory.GetFiles(directory, "*.fbx", SearchOption.AllDirectories);
            foreach (var file in meshFiles)
            {
                string name = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
                if (name.Contains("body") || name.Contains("base"))
                    _meshPaths[HumanLODLevel.LOD0_FullDetail] = file;
                else if (name.Contains("face"))
                    _faceMesh = null;
            }

            foreach (var lod in _lodConfigs)
            {
                string lodFile = Path.Combine(directory, $"lod{(int)lod.Level}.fbx");
                if (File.Exists(lodFile))
                    lod.MeshPath = lodFile;
            }
        }

        public void SaveToDirectory(string directory)
        {
            Directory.CreateDirectory(directory);
            var data = new HumanExportData
            {
                Name = _name,
                Gender = _gender,
                Ethnicity = _ethnicity,
                AgeGroup = _ageGroup,
                SkinTone = _skinTone,
                EyeColor = _eyeColor,
                HairStyle = _hairStyle,
                ClothingCount = _clothing.Count,
                AnimationCount = _animations.Count,
                MorphTargetGroups = new[]
                {
                    _faceExpressions.Targets.Count,
                    _faceShape.Targets.Count,
                    _bodyShape.Targets.Count
                }
            };
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(directory, "human.json"), json, Encoding.UTF8);
        }

        public string ExportForGameEngine(string outputPath, string format = "glTF")
        {
            var exportData = new
            {
                Name = _name,
                Gender = _gender.ToString(),
                Ethnicity = _ethnicity.ToString(),
                AgeGroup = _ageGroup.ToString(),
                Skeleton = _skeleton.Bones.Select(b => new { b.Name, b.ParentIndex, Region = b.Region.ToString() }).ToArray(),
                FaceMorphs = _faceExpressions.Targets.Select(t => new { t.Name, t.Category, t.CurrentValue }).ToArray(),
                FaceShapeMorphs = _faceShape.Targets.Select(t => new { t.Name, t.Category, t.CurrentValue }).ToArray(),
                BodyMorphs = _bodyShape.Targets.Select(t => new { t.Name, t.Category, t.CurrentValue }).ToArray(),
                Materials = new
                {
                    Skin = new { _skinMaterial.Name, Roughness = 0.55f, SSS = true },
                    Eye = new { _eyeMaterial.Name, IOR = 1.376f },
                    Hair = new { _hairMaterial.Name },
                    Teeth = new { _teethMaterial.Name },
                    Sclera = new { _scleraMaterial.Name }
                },
                Clothing = _clothing.Select(c => new { c.Name, Type = c.Type.ToString(), c.PrimaryColor }).ToArray(),
                LODs = _lodConfigs.Select(l => new { Level = l.Level.ToString(), l.ScreenSize, l.MaxMorphTargets, l.MeshPath }).ToArray()
            };
            var json = JsonSerializer.Serialize(exportData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(outputPath, json, Encoding.UTF8);
            return outputPath;
        }

        public SynapseHuman Clone(string newName = null)
        {
            var clone = new SynapseHuman(newName ?? $"{_name}_Clone")
            {
                _gender = _gender,
                _ethnicity = _ethnicity,
                _ageGroup = _ageGroup,
                _skinTone = _skinTone,
                _eyeColor = _eyeColor,
                _hairStyle = _hairStyle
            };
            clone.SetAppearance(_skinTone, _eyeColor, _hairStyle);
            foreach (var clothing in _clothing)
                clone.AddClothing(clothing);
            foreach (var anim in _animations)
                clone.AddAnimation(anim);
            return clone;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _clothing.Clear();
                _animations.Clear();
                _meshPaths.Clear();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

    internal class HumanExportData
    {
        public string Name { get; set; }
        public HumanGender Gender { get; set; }
        public HumanEthnicity Ethnicity { get; set; }
        public HumanAgeGroup AgeGroup { get; set; }
        public HumanSkinTone SkinTone { get; set; }
        public HumanEyeColor EyeColor { get; set; }
        public HumanHairStyle HairStyle { get; set; }
        public int ClothingCount { get; set; }
        public int AnimationCount { get; set; }
        public int[] MorphTargetGroups { get; set; }
    }

    // =========================================================================
    // HUMAN PRESET LIBRARY
    // =========================================================================

    public class SynapseHumanPresets
    {
        public static SynapseHuman CreateEuropeanFemale() =>
            new SynapseHuman("European Female").Also(h => h.SetAppearance(HumanSkinTone.Fair, HumanEyeColor.Blue, HumanHairStyle.Long, HumanGender.Female, HumanEthnicity.European, HumanAgeGroup.YoungAdult));

        public static SynapseHuman CreateEuropeanMale() =>
            new SynapseHuman("European Male").Also(h => h.SetAppearance(HumanSkinTone.Light, HumanEyeColor.Hazel, HumanHairStyle.Short, HumanGender.Male, HumanEthnicity.European, HumanAgeGroup.Adult));

        public static SynapseHuman CreateEastAsianFemale() =>
            new SynapseHuman("East Asian Female").Also(h => h.SetAppearance(HumanSkinTone.Light, HumanEyeColor.Brown, HumanHairStyle.Long, HumanGender.Female, HumanEthnicity.EastAsian, HumanAgeGroup.YoungAdult));

        public static SynapseHuman CreateEastAsianMale() =>
            new SynapseHuman("East Asian Male").Also(h => h.SetAppearance(HumanSkinTone.Light, HumanEyeColor.Brown, HumanHairStyle.Short, HumanGender.Male, HumanEthnicity.EastAsian, HumanAgeGroup.Adult));

        public static SynapseHuman CreateAfricanFemale() =>
            new SynapseHuman("African Female").Also(h => h.SetAppearance(HumanSkinTone.Dark, HumanEyeColor.Brown, HumanHairStyle.Afro, HumanGender.Female, HumanEthnicity.African, HumanAgeGroup.YoungAdult));

        public static SynapseHuman CreateAfricanMale() =>
            new SynapseHuman("African Male").Also(h => h.SetAppearance(HumanSkinTone.Ebony, HumanEyeColor.Brown, HumanHairStyle.BuzzCut, HumanGender.Male, HumanEthnicity.African, HumanAgeGroup.Adult));

        public static SynapseHuman CreateMiddleEasternFemale() =>
            new SynapseHuman("Middle Eastern Female").Also(h => h.SetAppearance(HumanSkinTone.Olive, HumanEyeColor.Brown, HumanHairStyle.Long, HumanGender.Female, HumanEthnicity.MiddleEastern, HumanAgeGroup.YoungAdult));

        public static SynapseHuman CreateSouthAsianMale() =>
            new SynapseHuman("South Asian Male").Also(h => h.SetAppearance(HumanSkinTone.Tan, HumanEyeColor.Brown, HumanHairStyle.Short, HumanGender.Male, HumanEthnicity.SouthAsian, HumanAgeGroup.Adult));

        public static SynapseHuman CreateElderlyMale() =>
            new SynapseHuman("Elderly Male").Also(h => h.SetAppearance(HumanSkinTone.Light, HumanEyeColor.Grey, HumanHairStyle.Bald, HumanGender.Male, HumanEthnicity.European, HumanAgeGroup.Elderly));
    }

    public static class SynapseHumanExtensions
    {
        public static T Also<T>(this T obj, Action<T> action) { action(obj); return obj; }
    }
}
