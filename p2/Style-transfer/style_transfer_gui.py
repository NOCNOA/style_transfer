import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                            QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                            QSpinBox, QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
from PIL import Image
import torchvision.transforms as transforms

from transfer_ import VGGNet, load_img, style_transfer


class StyleTransferGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.content_path = None
        self.style_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_processing = False  # 添加处理状态标志
        self.img_size = 512
        
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
    def initUI(self):
        self.setWindowTitle('风格迁移工具')
        self.setGeometry(100, 100, 1400, 700)
        
        # 设置主窗口背景色
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #E8E8E8;")  # 浅灰色背景
        self.setCentralWidget(main_widget)
        
        # 主布局
        layout = QHBoxLayout() 
        layout.setSpacing(50)  
        


        # 左侧布局（内容图像）
        left_layout = QVBoxLayout()
        content_frame = QWidget()
        content_frame.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        content_frame_layout = QVBoxLayout()#内容布局
        self.content_label = QLabel('内容图像')
        self.content_label.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold;
            padding: 3px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
        """)
        self.content_label.setAlignment(Qt.AlignCenter)
        self.content_label.setFixedHeight(25)  # 固定高度
        self.content_image = QLabel()
        self.content_image.setFixedSize(400, 400)
        self.content_image.setStyleSheet("background-color: white;")
        self.content_btn = QPushButton('选择内容图像')
        self.content_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;  /* 深灰色 */
                color: white;
                padding: 5px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #616161;  /* 悬停时稍微亮一点的灰色 */
            }
        """)
        self.content_btn.clicked.connect(self.load_content_image)
        content_frame_layout.addWidget(self.content_label)
        content_frame_layout.addWidget(self.content_image)
        content_frame_layout.addWidget(self.content_btn)
        content_frame.setLayout(content_frame_layout)
        left_layout.addWidget(content_frame)
        
        # 中间布局（风格图像）
        middle_layout = QVBoxLayout()
        style_frame = QWidget()
        style_frame.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        style_frame_layout = QVBoxLayout()
        self.style_label = QLabel('风格图像')
        self.style_label.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold;
            padding: 3px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
        """)
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_label.setFixedHeight(25)  # 固定高度
        self.style_image = QLabel()
        self.style_image.setFixedSize(400, 400)
        self.style_image.setStyleSheet("background-color: white;")
        self.style_btn = QPushButton('选择风格图像')
        self.style_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;  /* 深灰色 */
                color: white;
                padding: 5px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #616161;  /* 悬停时稍微亮一点的灰色 */
            }
        """)
        self.style_btn.clicked.connect(self.load_style_image)
        style_frame_layout.addWidget(self.style_label)
        style_frame_layout.addWidget(self.style_image)
        style_frame_layout.addWidget(self.style_btn)
        style_frame.setLayout(style_frame_layout)
        middle_layout.addWidget(style_frame)
        
        # 右侧布局（结果图像）
        right_layout = QVBoxLayout()
        result_frame = QWidget()
        result_frame.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        result_frame_layout = QVBoxLayout()
        self.result_label = QLabel('结果图像')
        self.result_label.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold;
            padding: 3px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedHeight(25)  # 固定高度
        self.result_image = QLabel()
        self.result_image.setFixedSize(400, 400)
        self.result_image.setStyleSheet("background-color: white;")
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BDBDBD;
                border-radius: 3px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #424242;
            }
        """)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()  # 初始时隐藏进度条
        
        # 添加迭代次数选择
        iter_layout = QHBoxLayout()
        iter_label = QLabel('迭代次数:')
        self.iter_input = QSpinBox()
        self.iter_input.setRange(10, 50000)  
        self.iter_input.setValue(2000)       
        self.iter_input.setSingleStep(100)   
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.iter_input)
        
        # 添加工具提示
        self.iter_input.setToolTip("""
            迭代次数说明：
            - 500-1000: 快速预览效果
            - 1000-2000: 一般效果
            - 2000-3000: 较好效果
            - 3000以上: 精细效果
            注意：迭代次数越多，处理时间越长
        """)
        
        self.transfer_btn = QPushButton('开始转换')
        self.transfer_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                padding: 5px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        self.transfer_btn.clicked.connect(self.toggle_transfer)
        
        result_frame_layout.addWidget(self.result_label)
        result_frame_layout.addWidget(self.result_image)
        result_frame_layout.addWidget(self.progress_bar)  # 添加进度条
        result_frame_layout.addLayout(iter_layout)
        result_frame_layout.addWidget(self.transfer_btn)
        result_frame.setLayout(result_frame_layout)
        right_layout.addWidget(result_frame)
        
        # 将三个布局添加到主布局
        layout.addLayout(left_layout)
        layout.addLayout(middle_layout)
        layout.addLayout(right_layout)
        
        main_widget.setLayout(layout)
        
        # 修改迭代次数输入框样式
        self.iter_input.setStyleSheet("""
            QSpinBox {
                background-color: white;
                border: 1px solid #BDBDBD;
                border-radius: 3px;
                padding: 2px;
            }
        """)
    
    def load_content_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择内容图像', '', 
                                             'Image files (*.jpg *.jpeg *.png)')
        if fname:
            self.content_path = fname
            pixmap = QPixmap(fname)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
            self.content_image.setPixmap(scaled_pixmap)
    
    def load_style_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择风格图像', '', 
                                             'Image files (*.jpg *.jpeg *.png)')#选择风格图像
        if fname:
            self.style_path = fname
            pixmap = QPixmap(fname) #将选择风格图像转换为QPixmap
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio) 
            self.style_image.setPixmap(scaled_pixmap)
    
    def toggle_transfer(self):
        """切换开始/停止状态"""
        if not self.is_processing:
            # 开始处理
            self.start_transfer()
        else:
            # 停止处理
            self.stop_transfer()
    
    def start_transfer(self):
        if not self.content_path and not self.style_path:
            QMessageBox.warning(self, '提示', 
                              '请先选择内容图像和风格图像！',
                              QMessageBox.Ok)
            return
        elif not self.content_path:
            QMessageBox.warning(self, '提示', 
                              '请先选择内容图像！',
                              QMessageBox.Ok)
            return
        elif not self.style_path:
            QMessageBox.warning(self, '提示', 
                              '请先选择风格图像！',
                              QMessageBox.Ok)
            return
            
        # 确认开始
        reply = QMessageBox.question(self, '确认', 
                                   '确定要开始处理吗？\n处理过程可能需要几分钟。',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
            
        total_step = self.iter_input.value()
        
        # 设置处理状态
        self.is_processing = True
        self.transfer_btn.setText('停止转换')
        self.transfer_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                padding: 5px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
        """)
        
        # 初始化进度条
        self.progress_bar.setMaximum(total_step)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # 禁用其他按钮
        self.content_btn.setEnabled(False)
        self.style_btn.setEnabled(False)
        self.iter_input.setEnabled(False)
        
        try:
            # 调用风格迁移函数
            result_tensor = style_transfer(
                content_path=self.content_path,
                style_path=self.style_path,
                total_step=total_step,
                callback=self.update_result_image,
                progress_callback=self.update_progress,
                stop_flag=lambda: not self.is_processing  # 添加停止检查
            )
            
            if self.is_processing:  # 只有在正常完成时才保存和提示
                self.save_final_result(result_tensor)
                QMessageBox.information(self, '提示', 
                                      f'处理完成！\n结果已保存至: output 文件夹',
                                      QMessageBox.Ok)
        finally:
            # 恢复界面状态
            self.reset_ui_state()
    
    def stop_transfer(self):
        """停止处理"""
        reply = QMessageBox.question(self, '确认', 
                                   '确定要停止处理吗？\n当前进度将丢失。',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.is_processing = False
    
    def reset_ui_state(self):
        """重置界面状态"""
        self.is_processing = False
        self.transfer_btn.setText('开始转换')
        self.transfer_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                padding: 5px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        self.content_btn.setEnabled(True)
        self.style_btn.setEnabled(True)
        self.iter_input.setEnabled(True)
        self.progress_bar.hide()
    
    def update_result_image(self, tensor):
        # 将tensor转换为PIL图像
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = tensor.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
        img = transforms.ToPILImage()(img)
        
        # 转换为QPixmap并显示
        img = img.convert('RGB')
        data = img.tobytes('raw', 'RGB')
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.result_image.setPixmap(scaled_pixmap)
        QApplication.processEvents()
    
    def save_final_result(self, tensor):
        """保存最终结果（带时间戳）"""
        # 获取文件名
        content_name = os.path.splitext(os.path.basename(self.content_path))[0]
        style_name = os.path.splitext(os.path.basename(self.style_path))[0]
        iterations = self.iter_input.value()
        
        # 获取当前时间
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 构建新的文件名
        output_name = f"{style_name}_{content_name}_{iterations}steps_{timestamp}.png"
        
        # 确保输出路径存在
        output_path = os.path.join('output', output_name)
        
        # 转换并保存图像
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = tensor.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
        img = transforms.ToPILImage()(img)
        
        # 保存图像
        img.save(output_path)
        print(f'结果已保存至: {output_path}')
    
    def update_progress(self, step):
        """更新进度条"""
        self.progress_bar.setValue(step)
        QApplication.processEvents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StyleTransferGUI()
    ex.show()
    sys.exit(app.exec_()) 