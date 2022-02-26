import bpy


def renderImage(path):
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport = True, write_still=True)
    
# bpy.context.scene.camera = bpy.data.objects["Camera"]

#for ob in bpy.data.objects:
#    if ob.type == 'CAMERA':
#        print(ob.name)

def generateCam(num):
    for i in range(num):
        bpy.ops.object.camera_add(
            enter_editmode=False, 
            align='VIEW', 
            location=(30, -20, 30), 
            rotation=(1.5708, 0, 1.5708),
            scale=(1, 1, 1))
        
        name = 'Camera' + str(i+1)
        bpy.context.object.name = name

