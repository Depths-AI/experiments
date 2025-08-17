from numba import njit
import numpy as np

@njit(nogil=True)
def heap_push(heap_distances, heap_indices, dist, index, pos):
    """
    Pushes a new item onto the max-heap at a specific position and sifts up.
    The heap is represented by two arrays: one for distances and one for indices.
    """
    # Add the new item to the end of the heap
    heap_distances[pos] = dist
    heap_indices[pos] = index
    
    # Sift up to maintain the heap property
    while pos > 0:
        parent_pos = (pos - 1) // 2
        if heap_distances[pos] > heap_distances[parent_pos]:
            # Swap with parent
            heap_distances[pos], heap_distances[parent_pos] = heap_distances[parent_pos], heap_distances[pos]
            heap_indices[pos], heap_indices[parent_pos] = heap_indices[parent_pos], heap_indices[pos]
            pos = parent_pos
        else:
            break

@njit(nogil=True)
def heap_replace(heap_distances, heap_indices, dist, index):
    """
    Replaces the largest item in the max-heap and sifts down.
    This is more efficient than a pop followed by a push.
    """
    # Replace the root of the heap
    heap_distances[0] = dist
    heap_indices[0] = index
    
    # Sift down to maintain the heap property
    pos = 0
    k = heap_distances.shape[0]
    while True:
        left_child = 2 * pos + 1
        right_child = 2 * pos + 2
        largest = pos
        
        if left_child < k and heap_distances[left_child] > heap_distances[largest]:
            largest = left_child
        
        if right_child < k and heap_distances[right_child] > heap_distances[largest]:
            largest = right_child
        
        if largest != pos:
            # Swap with the largest child
            heap_distances[pos], heap_distances[largest] = heap_distances[largest], heap_distances[pos]
            heap_indices[pos], heap_indices[largest] = heap_indices[largest], heap_indices[pos]
            pos = largest
        else:
            break
