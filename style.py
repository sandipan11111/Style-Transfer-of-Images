from helper import *
import tensorflow as tf
import numpy as np
import vgg16

def style_transfer(content_image,style_image,content_layer_ids,style_layer_ids,weight_content=1.5,weight_style=10.0,weight_denoise=0.3,num_iterations=120,step_size=10.0):

	model=vgg16.VGG16()
	session=tf.InteractiveSessionn(graph=model.graph)

	#create the different losses
	loss_content=content_loss(session=session,model=model,content_image=content_image,layer_ids=content_layer_ids)

	loss_style=style_loss(session=session,model=model,content_image=content_image,layer_ids=style_layer_ids)

	loss_denoise=create_denoise_loss(model)

	adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')


	#Initialize the session
	session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

	##Reciprocal values with handling of denominator 0 case
	update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))


	#Total Loss
	loss_combined = weight_content * adj_content * loss_content +
                    weight_style * adj_style * loss_style +
                    weight_denoise * adj_denoise * loss_denoise


	#define gradient
	gradient = tf.gradients(loss_combined, model.input)
	run_list = [gradient, update_adj_content, update_adj_style,
                update_adj_denoise]


	#Our result/mixed image
	 mixed_image = np.random.rand(*content_image.shape) + 128

	 #Gradient Descent
	 for i in range(num_iterations):

        feed_dict = model.create_feed_dict(image=mixed_image)
		grad, adj_content_val, adj_style_val, adj_denoise_val
        = session.run(run_list, feed_dict=feed_dict)
        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        mixed_image -= grad * step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        print(". ", end="")
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)

    print()
    print("Final image:")
    plot_image_big(mixed_image)
    session.close()
