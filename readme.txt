Soru 1-) Merhaba öncelikle tool olarak Detectron2 ve içinde yapı olarakta Fast R-CNN kullandığımı belirtmek isterim.

Eğitim amaçlı kullandığımı model input verileri ise şunlardır ;

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.GAMMA = 0.05
cfg.TEST.EVAL_PERIOD = 200
cfg.SOLVER.MAX_ITER = 2000    
cfg.SOLVER.STEPS = [500]        
   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

Elde ettiğim diğer veriler ise dosya içinde mevcuttur.

Soru 2-) Ben aslında araştırma konularının ilerledikçe geliştirme alanınında buna paralel olarak ilerleneceğini düşünüyorum. Çünkü her ne kadar kodlarla
ve uygulamalarla ilerlenilse de bu bir bilimdir ve arkasında belli matematiksel veriler içermektedir. Endüstriyel olarakta yapılan uygulamaların çok az 
bir kısmının sahada uygulandığını ve bununla birlikte yeni şeylerin denenmeye açık olması gerektiğini düşünüyorum. 