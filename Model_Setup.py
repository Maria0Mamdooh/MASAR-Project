!sed -n '1,200p' model.py

!grep -R "vit\|ViT\|transformer\|TransGeo" .

%cd /content
!git clone https://github.com/jeff-zilence/transgeo2022.git
%cd transgeo2022
!ls

!sed -n '1,200p' train.py

!sed -n '1,120p' model/TransGeo.py

!find . -name "*.pth" -o -name "*.pt" -o -name "*.tar"

%cd /content/transgeo2022
!ls

!pip install timm einops ptflops faiss-cpu

transgeo_path = "model/TransGeo.py"

with open(transgeo_path, "r") as f:
    code = f.read()

old_text = """        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]"""

new_text = """        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]
        elif args.dataset == 'university':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [256, 256]"""

if "args.dataset == 'university'" not in code:
    code = code.replace(old_text, new_text)

    with open(transgeo_path, "w") as f:
        f.write(code)

    print("Added university dataset settings to TransGeo.py")
else:
    print("university settings already exist")

!pip uninstall sympy -y
!pip install sympy==1.13.3

%cd /content/transgeo2022
!ls

