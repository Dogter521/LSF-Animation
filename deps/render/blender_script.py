import bpy
import numpy as np
import os, sys

filename = str(sys.argv[-1])
root_dir = str(sys.argv[-2])

seq_pickle = np.load(root_dir + filename + '.npy')

seq_pickle = np.squeeze(seq_pickle, axis=0)

shape = seq_pickle[:, :300]
exp = seq_pickle[:, 300:400]
jaw = seq_pickle[:, 400:]


bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'MATCAP'
bpy.context.scene.display.render_aa = 'FXAA'
bpy.context.scene.render.resolution_x = int(1280)
bpy.context.scene.render.resolution_y = int(720)
bpy.context.scene.render.fps = 25
bpy.context.scene.render.image_settings.file_format = 'PNG'

cam = bpy.data.objects['Camera']
bpy.context.scene.camera = cam

frames = exp.shape[0]

flame_obj = bpy.context.scene.objects["FLAME2020-generic"] 

flame_obj.select_set(True)

output_dir = root_dir + filename
for frame in range(frames):
    keyBlocks = bpy.data.shape_keys[0].key_blocks
    
    for count in range(1,301,1):
        keyBlocks[count].value = shape[frame][count-1]
#        keyBlocks[count].keyframe_insert("value", frame=frame)
        
    for count in range(301,401,1):
        keyBlocks[count].value = exp[frame][count-301]
#        keyBlocks[count].keyframe_insert("value", frame=frame)
        
    bpy.ops.object.posemode_toggle()
    jaw_bone = flame_obj.pose.bones.get("jaw")
    jaw_bone.rotation_mode = 'XYZ'
    jaw_bone.rotation_euler = (jaw[frame][0], jaw[frame][1], jaw[frame][2])
#    jaw_bone.keyframe_insert("rotation_euler", frame=frame)
    bpy.ops.object.posemode_toggle()
    bpy.context.scene.render.filepath = os.path.join(output_dir, '{}.png'.format(frame))
    bpy.ops.render.render(write_still=True)
