import vtk
import numpy as num
from vtk.util import numpy_support
from qtest import invert


def numpy_to_vtk(a):
    flattened = a.flatten(order='F')
    data = numpy_support.numpy_to_vtk(flattened, deep=True)
    data.SetNumberOfComponents(3)
    return data


def vtk_grid(pnts, xg, yg, zg, normalize=True):
    '''
    :param pnts: matrix of size (len(xg), len(yg), len(zg)) with scales
    :param xg, yg, zg: coordinates vectors.'''
    actors = []
    if normalize:
        pntsmax = num.max(pnts)
    else:
        pntsmax = 1.

    for ix, xi in enumerate(xg):
        for iy, yi in enumerate(yg):
            for iz, zi in enumerate(zg):
                scale = 0.2 + 4.*(pnts[ix, iy, iz]/pntsmax)
                actors.append(
                    vtk_point(num.array((xi, yi, zi)),
                                        scale=scale))
    return actors


def grid_actors(pnts, xg, yg, zg, normalize=True):
    '''
    :param pnts: matrix of size (len(xg), len(yg), len(zg)) with scales
    :param xg, yg, zg: coordinates vectors.

    grid needs to be regularly spaced!'''
    actors = []
    pnts = num.copy(pnts)
    if normalize:
        pnts -= num.min(pnts)
        pntsmax = num.max(pnts)
    else:
        pntsmax = 1.

    pnts /= pntsmax
    opacities = pnts/10.
    opacities[num.ma.getmaskarray(opacities)] = 0

    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    dz = zg[1] - zg[0]

    for ix, xi in enumerate(xg):
        for iy, yi in enumerate(yg):
            for iz, zi in enumerate(zg):
                s = vtk.vtkCubeSource()
                s.SetCenter(xi, yi, zi)
                s.SetXLength(dx)
                s.SetYLength(dy)
                s.SetZLength(dz)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(s.GetOutputPort())

                actor = vtk.vtkActor()
                actor.GetProperty().SetOpacity(opacities[ix, iy, iz])
                actor.SetMapper(mapper)
                actors.append(actor)
    return actors



def vtk_points(pnts, cube_size=10):
    '''pnts: numpy array of length (3, X) '''
    data = numpy_to_vtk(pnts)
    s = vtk.vtkPoints()
    s.SetData(data)

    cs = vtk.vtkCubeSource()
    cs.SetXLength(cube_size)
    cs.SetYLength(cube_size)
    cs.SetZLength(cube_size)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(s)

    # the glyph filter giving shape to the points:
    g = vtk.vtkGlyph3D()
    g.SetScaleModeToDataScalingOff()
    g.SetInput(polydata)
    g.SetSource(cs.GetOutput())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(g.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(100)

    return actor


def vtk_point(pnts, scale=1):
    '''pnts: numpy array of length 3

    actually, it's a sphere not a point'''
    data = numpy_to_vtk(pnts)
    s = vtk.vtkSphereSource()
    s.SetCenter(*pnts)
    s.SetRadius(20.*scale)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(s.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def vtk_ray(x, opacity=1.):
    '''pnts: numpy array of shape (3, X) for X is number of points.
    So, it's x,y,z columns'''
    if isinstance(x, invert.Ray3D):
        pnts = num.array(x.nezt[:3])
    else:
        pnts = x
    data = numpy_to_vtk(pnts)
    points = vtk.vtkPoints()
    points.SetData(data)

    lines = vtk.vtkCellArray()
    npoints = pnts.shape[1]
    lines.InsertNextCell(npoints)
    for i in range(npoints):
        lines.InsertCellPoint(i)

    polygon = vtk.vtkPolyData()
    polygon.SetPoints(points)
    polygon.SetLines(lines)

    # standard stuff:
    polygonMapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonMapper.SetInputConnection(polygon.GetProducerPort())
    else:
        polygonMapper.SetInputData(polygon)
        polygonMapper.Update()

    polygonActor = vtk.vtkActor()
    polygonActor.GetProperty().SetPointSize(20)
    p = polygonActor.GetProperty()
    p.SetPointSize(20)
    p.SetOpacity(opacity)

    polygonActor.GetProperty().SetPointSize(20)
    polygonActor.SetMapper(polygonMapper)

    return polygonActor

def render_actors(actors):
    ''' Take a list of actors and render them.'''
    ren1 = vtk.vtkRenderer()
    for a in actors:
        ren1.AddActor(a)
    ren1.SetBackground(0.1, 0.2, 0.4)
    ren1.ResetCamera()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.SetSize(300, 300)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


#__all__ = [vtk_ray, render_actors]
