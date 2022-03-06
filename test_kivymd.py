*a,b,*c=1,2,3,5,7,8,5,4,7,8
print(a,b,c)
MDTextField:
id: data_FD
size_hint_x: None
width: 250
hint_text: "FD Кол-во отсчетов в секунду"
mode: "fill"
on_text: app.get_FD(root.ids.data_FD.text)