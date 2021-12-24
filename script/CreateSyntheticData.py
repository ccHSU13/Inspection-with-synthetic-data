import bpy
import random
import os
from mathutils import Vector


def getTexturePath() :
    abs_path = os.path.realpath(__file__)
    return abs_path[:abs_path.rfind("Bridge_3")] + "DefectTexture"


# ========================================================================
# Use for set MANUALLY planes onto the bridge. 
# 1) Set the cursor according to the geometry, 2) Run this section

def createPlane():
    s_x = random.uniform(1, 5)
    s_y = random.uniform(0.5, 5)

    plane = bpy.ops.mesh.primitive_plane_add(size=0.5, align='CURSOR')
    bpy.context.object.scale[0] = s_x
    bpy.context.object.scale[1] = s_y

    f = bpy.context.object.data.polygons[0]
    norm = f.normal
    norm  = norm @ bpy.context.scene.cursor.matrix
    Vector.normalize(norm)

    l = bpy.context.object.location 
    bpy.context.object.location = l - 0.05*norm

# ========================================================================


# ========================================================================
# Use to import texture, create defect material, and add materials onto planes
def generateTexture():
    path = getTexturePath()
    defects = os.listdir(path)

    for defect in defects:
        mat = bpy.data.materials.new(name = defect[:-4])
        mat.use_nodes = True

        bpy.context.object.active_material = mat
        matnodes = mat.node_tree.nodes

        principle_node = matnodes.get('Principled BSDF')
        tex = matnodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(path+'/'+defect)
        mat.node_tree.links.new(tex.outputs["Color"], principle_node.inputs['Base Color'])
        
    print("Finish generating defect textures")


def setMatOntoSelectedPlanes():
    mats = []
    for mat in bpy.data.materials:
        if 'Mask' not in mat.name:
            if 'Cr' in mat.name:
                mats.append(mat)
            if 'Ef' in mat.name:
                mats.append(mat)
            if 'Sp' in mat.name:
                mats.append(mat)
            if 'ER' in mat.name:
                mats.append(mat)
    
    random.shuffle(mats)
            
    planes = bpy.data.objects
    for p in planes:
        if p.type == 'MESH' and "plane" in p.name:
            n = random.randint(1,len(mats)-1)
            p.active_material = mats[n]
    
    print("Finish attaching texture on planes")
        

# =========================================================================

def displayObjects(hide=True):
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            if 'Layer' in ob.name or 'Object' in ob.name:
                ob.hide_render = hide
                ob.hide_set(hide)
    
    roughness = 0.5
    if (hide):
        roughness = 0.0
        
    bpy.data.materials["Sky"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = roughness
    bpy.data.materials["Ground"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = roughness


def assignMask():
    masks = {
        'Mask_Cr' : 0,
        'Mask_Ef' : 1,
        'Mask_Sp' : 2,
        'Mask_ER-Sp' : 3
    }

    materials = [None, None, None, None]
    for mat in bpy.data.materials:
        if mat.name in masks:
            materials[masks[mat.name]] = mat

    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            if 'Sp_ER' in ob.active_material.name:
                ob.active_material = materials[3]
                
            elif 'Cr' in ob.active_material.name:
                ob.active_material = materials[0]
            
            elif 'Sp' in ob.active_material.name:
                ob.active_material = materials[2]
            
            elif 'Ef' in ob.active_material.name:
                ob.active_material = materials[1]


#generateTexture()
#setMatOntoSelectedPlanes()

displayObjects()
assignMask()


