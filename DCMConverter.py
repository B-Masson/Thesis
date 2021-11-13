from posixpath import join
import dicom2nifti
print("Import successful.")
import os
import os.path as op

id = "I379425"
dicom_directory_pure = "AD_Subset/011_S_4845/AXIAL_T2_STAR/2013-07-08_07_30_28.0/" + id
dicom_directory = "AD_Subset"
print("Dicom directory exists:", op.exists(dicom_directory))
output_folder = "NIFTI_Test"
print("Output directory exists:", op.exists(output_folder))
for sample in os.listdir(dicom_directory):
    for type in os.listdir(op.join(dicom_directory, sample)):
        if ("AXIAL_T2_STAR" in type):
            print("Match in", sample)
            for case in os.listdir(op.join(dicom_directory, sample, type)):
                for image in os.listdir(op.join(dicom_directory, sample, type, case)):
                    path = op.join(dicom_directory, sample, type, case, image)
                    dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
        else:
            print("Rejected.")
print("Done.")