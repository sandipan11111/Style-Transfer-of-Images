import numpy as np
import tensorflow as tf 
import PIL.Image



def load_image(filename,max_size=None):
	image=PIL.Image.open(filename)

	if max_size is not None:

		factor=max_size/np.max(image.size)
		size=np.array(image.size)*factor
		size=size.astype(int)

		image=image.resize(size,PIL.Image.LANCZOS)

	return np.float32(image)


def save_image(image,filename):
	image=np.clip(image,0.0,255.0)
	image==image.astype(np.uint8)

	with open(filename,'wb') as file:
		PIL.Image.formarray(image).save(file,'jpeg')


def mean_squared_error(a,b):
	return tf.reduce_mean(tf.square(a-b))


def content_loss(session,model,content_image,layer,ids):
	feed_dict=model.create_feed_dict(image=content_image)
	layers=model.get_layer_tensors(layer_ids)
	values=session.run(layers,feed_dict=feed_dict)

	with model.graph.as_default():
		layer_losses=[]
		for value,layer in zip(values,layers):
			value_const=tf.constant(value)
			loss=mean_squared_error(layer,value_const)
			layer_losses.append(loss)

		total_loss=tf.reduce_mean(layer_losses)

	return total_loss


def gram_matrix(tensor):

	shape=tensor.get_shape()
	num_channels=int(shape[3])
	matrix=tf.reshape(tensor,shape=[-1,num_channels])
	gram=tf.matmul(tf.transpose(matrix),matrix)

	return gram


def style_loss(session,model,style_image,layer_ids):
	feed_dict=model.create_feed_dict(image=style_image)
	layers=model.get_layer_tensors(layer_ids)

	with model.graph.as_default():
		gram_layers=[gram_matrix(layer) for layer in layers]

		values=session.run(gram_layers,feed_dict=feed_dict)
		layer_losses=[]

		for value,gram_layer in zip(values,gram_layers):
			value_const=tf.constant(value)
			loss=mean_squared_error(gram_layer,value_const)
			layer_losses.append(loss)

		total_loss=tf.reduce_mean(layer_losses)

	return total_loss


def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss

