template <class T> 
class SoupVec {
    private:
        int len;
        T * vec;

    public:
        SoupVec();
        SoupVec(int l);
        SoupVec(int l, const T &v);
        SoupVec(int l, const T *v);

        SoupVec(const SoupVec &rv);
        SoupVec & operator = (const SoupVec &rv);

        typedef T vecType;

        inline T & operator[](const int i);
        inline const T & operator[](const int i) const;

        inline int size() const;
        void resize(int newlen);
        void assign(int newlen, const T &v);

        ~SoupVec();
};

template <class T> 
class SoupMat {
    private:
        int nrow;
        int ncol;
        T ** mat;

    public:
        SoupMat();
        SoupMat(int nr, int nc);
        SoupMat(int nr, int nc, const T &v);
        SoupMat(int nr, int nc, const T *v);

        SoupMat(const SoupMat &rm);
        SoupMat & operator = (const SoupMat &rm);

        typedef T matType;

        inline T* operator[](const int i);
        inline const T* operator[](const int i) const;
        inline int nRow() const;
        inline int nCol() const;
        void resize(int nr, int nc);
        void assign(int nr, int nc, const T &a);
        ~SoupMat();
};

//class Vector implementation

template <class T> SoupVec<T>::SoupVec(){
    len = 0;
}

template <class T> SoupVec<T>::SoupVec(int l){
    len = l;
    if (len > 0) vec = new T[len];
}

template <class T> SoupVec<T>::SoupVec(int l, const T &v){
    len = l;
    if (len>0){
        vec = new T[len];
        for(int i=0; i<len; i++) vec[i] = v;
    }
}

template <class T> SoupVec<T>::SoupVec(int l, const T *v){
    len = l;
    if (len>0) {
        vec = new T[len];
        for(int i=0; i<len; i++) vec[i] = *v++;
    }
}

template <class T> SoupVec<T>::SoupVec(const SoupVec &rv){
    len = rv.size();
    if (len > 0){
        vec = new T[len];
        for(int i=0; i<len; i++) vec[i] = rv[i];
    }
}

template <class T> SoupVec<T> & SoupVec<T>::operator=(const SoupVec<T>  &rv){
    int l = rv.size();
    if (l!=len){
        len = l;
        delete [] vec;
        vec = new T[len];
    }
    if (len > 0) for(int i=0; i<len; i++) vec[i] = rv[i];
    return *this;
}
    
template <class T> inline T & SoupVec<T>::operator[](const int i){
    return vec[i];
}

template <class T> inline const T & SoupVec<T>::operator[](const int i) const{
    return vec[i];
}

template <class T> inline int SoupVec<T>::size() const{
    return len;
}

template <class T> void SoupVec<T>::resize(int newlen){
    len = newlen;
    delete [] vec;
    if (len > 0) vec = new T[len];
}

template <class T> void SoupVec<T>::assign(int newlen, const T &v){
    len = newlen;
    delete [] vec;
    if (len > 0){
        vec = new T[len];
        for(int i=0;i<len;i++) vec[i] = v;
    }
}

template <class T> SoupVec<T>::~SoupVec(){
    delete [] vec;
}

//class Matrix implementation
template <class T> SoupMat<T>::SoupMat(){
    nrow = 0;
    ncol = 0;
}

template <class T> SoupMat<T>::SoupMat(int nr, int nc){
    nrow = nr;
    ncol = nc;
    if (nrow * ncol > 0) {
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol; 
    }
}

template <class T> SoupMat<T>::SoupMat(int nr, int nc, const T &v){
    nrow = nr;
    ncol = nc;
    if (nrow * ncol > 0) {
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
        for (int i = 0; i< nrow; i++) for (int j = 0; j<ncol; j++) mat[i][j]=v;
    }
}

template <class T> SoupMat<T>::SoupMat(int nr, int nc, const T *v){
    nrow = nr;
    ncol = nc;
    if (nrow * ncol > 0) {
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
        for (int i = 0; i< nrow; i++) for (int j = 0; j<ncol; j++) mat[i][j]=*v++;
    }
}

template <class T> SoupMat<T>::SoupMat(const SoupMat &rm){
    nrow = rm.nRow();
    ncol = rm.nCol();
    if (nrow * ncol > 0) {
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
        for (int i = 0; i< nrow; i++) for (int j = 0; j<ncol; j++) mat[i][j]=rm[i][j];
    }
}

template <class T> SoupMat<T> & SoupMat<T>::operator=(const SoupMat<T> &rm){
    int nr = rm.nRow();
    int nc = rm.nCol();
    if (nr * nc > 0) {
        if (nc != ncol || nr != nrow){
            nrow = nr;
            ncol = nc;
            mat = new T*[nrow];
            mat[0] = new T[nrow*ncol];
            for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
        }
        for (int i = 0; i< nrow; i++) for (int j = 0; j<ncol; j++) mat[i][j]=rm[i][j];
    }
    return *this;
}

template <class T> inline T* SoupMat<T>::operator[](const int i){
    return mat[i];
}

template <class T> inline const T* SoupMat<T>::operator[](const int i) const{
    return mat[i];
}

template <class T> inline int SoupMat<T>::nRow() const{
    return nrow;
}

template <class T> inline int SoupMat<T>::nCol() const{
    return ncol;
}

template <class T> void SoupMat<T>::resize(int nr, int nc){
    if(nr != nrow || nc != ncol){
        nrow = nr;
        ncol = nc;
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
    }
}

template <class T> void SoupMat<T>::assign(int nr, int nc, const T &a){
    if(nr != nrow || nc != ncol){
        nrow = nr;
        ncol = nc;
        mat = new T*[nrow];
        mat[0] = new T[nrow*ncol];
        for (int i = 1; i< nrow; i++) mat[i] = mat[i-1] + ncol;
    }
    for (int i = 0; i<nrow; i++) for (int j=0; j<ncol; j++) mat[i][j] = a;
}

template <class T> SoupMat<T>::~SoupMat(){
    delete [] mat[0];
    delete [] mat;
}
