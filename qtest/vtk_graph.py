import vtk
from vtk.util import numpy_support


def numpy_to_vtk(a):
    npoints = a.shape[1]
    flattened = a.flatten(order='F')
    data = numpy_support.numpy_to_vtk(flattened, deep=True)
    data.SetNumberOfComponents(3)
    return data

def vtk_point(pnts):
    data = numpy_to_vtk(pnts)
    s = vtk.vtkSphereSource()
    s.SetCenter(*pnts)
    s.SetRadius(20.)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(s.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

def vtk_ray(pnts, opacity=1.):
    #print 'RAY', pnts
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
